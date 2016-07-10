# -*- coding: utf-8 -*-
__author__ = 'raphaelfettaya'
import copy
import regex as re
# from dateutil import tz, parser
import dateparser

class DateFinder(object):
    """
    Locates dates in a text
    """

    DIGITS_MODIFIER_PATTERN = '\d+st|\d+th|\d+rd|first|second|third|fourth|fifth|sixth|seventh|eighth|nineth|tenth|next|last'
    DIGITS_PATTERN = '\d+'
    DAYS_PATTERN = 'lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche'
    MONTHS_PATTERN = "janvier|février|fevrier|mars|avril|mai|juin|juillet|aout|août|septembre|octobre|novembre|decembre"
    TIMEZONES_PATTERN = ''
    ## explicit north american timezones that get replaced
    NA_TIMEZONES_PATTERN = ''
    ALL_TIMEZONES_PATTERN = ''
    DELIMITERS_PATTERN = '[/\:\-\,\s\_\+\@]+'
    TIME_PERIOD_PATTERN = 'a\.m\.|am|p\.m\.|pmh'
    ## can be in date strings but not recognized by dateutils
    EXTRA_TOKENS_PATTERN = "dans|à|h|le|vers|pour|de|a"

    ## TODO: Get english numbers?
    ## http://www.rexegg.com/regex-trick-numbers-in-english.html

    RELATIVE_PATTERN = 'avant|après|prochain|dernier|ago'
    TIME_SHORTHAND_PATTERN = "midi|minuit|aujourd'hui|ojrd|hier|demain|après-demain|tomorrow"
    UNIT_PATTERN = 'seconde|minute|heures|jour|semaine|mois|année'

    ## Time pattern is used independently, so specified here.
    TIME_PATTERN = """
    (?P<time>
        ## Captures in format XX:YY(:ZZ) (PM) (EST)
        (
            (?P<hours>\d{{1,2}})
            \:
            (?P<minutes>\d{{1,2}})
            (\:(?<seconds>\d{{1,2}}))?
            \s*
            (?P<time_periods>{time_periods})?
            \s*
            (?P<timezones>{timezones})?
        )
        |
        ## Captures in format 11 AM (EST)
        ## Note with single digit capture requires time period
        (
            (?P<hours>\d{{1,2}})
            \s*
            (?P<time_periods>{time_periods})
            \s*
            (?P<timezones>{timezones})*
        )
    )
    """.format(
        time_periods=TIME_PERIOD_PATTERN,
        timezones=ALL_TIMEZONES_PATTERN
    )

    DATES_PATTERN = """
    (
        (
            {time}
            |
            ## Grab any digits
            (?P<digits_modifier>{digits_modifier})
            |
            (?P<digits>{digits})
            |
            (?P<days>{days})
            |
            (?P<months>{months})
            |
            ## Delimiters, ie Tuesday[,] July 18 or 6[/]17[/]2008
            ## as well as whitespace
            (?P<delimiters>{delimiters})
            |
            ## These tokens could be in phrases that dateutil does not yet recognize
            ## Some are US Centric
            (?P<extra_tokens>{extra_tokens})
        ## We need at least three items to match for minimal datetime parsing
        ## ie 10pm
        ){{1,}}
    )
    """

    DATES_PATTERN = DATES_PATTERN.format(
        time=TIME_PATTERN,
        digits=DIGITS_PATTERN,
        digits_modifier=DIGITS_MODIFIER_PATTERN,
        days=DAYS_PATTERN + '|' + TIME_SHORTHAND_PATTERN,
        relative=RELATIVE_PATTERN,
        months=MONTHS_PATTERN,
        delimiters=DELIMITERS_PATTERN,
        extra_tokens=EXTRA_TOKENS_PATTERN
    )

    DATE_REGEX = re.compile(DATES_PATTERN, re.IGNORECASE | re.MULTILINE | re.UNICODE | re.DOTALL | re.VERBOSE)

    TIME_REGEX = re.compile(TIME_PATTERN, re.IGNORECASE | re.MULTILINE | re.UNICODE | re.DOTALL | re.VERBOSE)

    ## These tokens can be in original text but dateutil
    ## won't handle them without modification
    REPLACEMENTS = {
        "standard": " ",
        "daylight": " ",
        "savings": " ",
        "time": " ",
        "date": " ",
        "by": " ",
        "due": " ",
        "on": " ",
        "to": " ",
    }

    TIMEZONE_REPLACEMENTS = {
        "pacific": "PST",
        "eastern": "EST",
        "mountain": "MST",
        "central": "CST",
    }

    ## Characters that can be removed from ends of matched strings
    STRIP_CHARS = ' \n\t:-.,_'

    def find_dates(self, text, source=False, index=False, strict=False):

        for date_string, indices, captures in self.extract_date_strings(text, strict=strict):

            as_dt = self.parse_date_string(date_string, captures)
            if as_dt is None:
                ## Dateutil couldn't make heads or tails of it
                ## move on to next
                continue

            returnables = (as_dt,)
            if source:
                returnables = returnables + (date_string,)
            if index:
                returnables = returnables + (indices,)

            if len(returnables) == 1:
                returnables = returnables[0]
            yield returnables

    def _find_and_replace(self, date_string, captures):
        """
        :warning: when multiple tz matches exist the last sorted capture will trump
        :param date_string:
        :return: date_string, tz_string
        """
        # add timezones to replace
        cloned_replacements = copy.copy(self.REPLACEMENTS)  # don't mutate
        for tz_string in captures.get('timezones', []):
            cloned_replacements.update({tz_string: ' '})

        date_string = date_string.lower()
        for key, replacement in cloned_replacements.items():
            # we really want to match all permutations of the key surrounded by whitespace chars except one
            # for example: consider the key = 'to'
            # 1. match 'to '
            # 2. match ' to'
            # 3. match ' to '
            # but never match r'(\s|)to(\s|)' which would make 'october' > 'ocber'
            date_string = re.sub(r'(^|\s)' + key + '(\s|$)', replacement, date_string, flags=re.IGNORECASE)

        return date_string, self._pop_tz_string(sorted(captures.get('timezones', [])))

    def _pop_tz_string(self, list_of_timezones):
        try:
            tz_string = list_of_timezones.pop()
            # make sure it's not a timezone we
            # want replaced with better abbreviation
            return self.TIMEZONE_REPLACEMENTS.get(tz_string, tz_string)
        except IndexError:
            return ''

    # def _add_tzinfo(self, datetime_obj, tz_string):
    #     """
    #     take a naive datetime and add dateutil.tz.tzinfo object
    #     :param datetime_obj: naive datetime object
    #     :return: datetime object with tzinfo
    #     """
    #     if datetime_obj is None:
    #         return None
    #
    #     tzinfo_match = tz.gettz(tz_string)
    #     return datetime_obj.replace(tzinfo=tzinfo_match)

    def parse_date_string(self, date_string, captures):
        # TODO clean that problem with demain
        date_string = date_string.replace("demain", "tomorrow")
        # replace tokens that are problematic for dateutil
        date_string, tz_string = self._find_and_replace(date_string, captures)

        ## One last sweep after removing
        date_string = date_string.strip(self.STRIP_CHARS)
        ## Match strings must be at least 3 characters long
        ## < 3 tends to be garbage
        if len(date_string) < 3:
            return None
        as_dt = dateparser.parse(date_string)
        # if tz_string:
        #     as_dt = self._add_tzinfo(as_dt, tz_string)
        return as_dt

    def extract_date_strings(self, text, strict=False):
        """
        Scans text for possible datetime strings and extracts them
        source: also return the original date string
        index: also return the indices of the date string in the text
        strict: Strict mode will only return dates sourced with day, month, and year
        """
        for match in self.DATE_REGEX.finditer(text):
            match_str = match.group(0)
            indices = match.span(0)

            ## Get individual group matches
            captures = match.capturesdict()
            time = captures.get('time')
            digits = captures.get('digits')
            digits_modifiers = captures.get('digits_modifiers')
            days = captures.get('days')
            months = captures.get('months')
            timezones = captures.get('timezones')
            delimiters = captures.get('delimiters')
            time = captures.get('time)')
            time_periods = captures.get('time_periods')
            extra_tokens = captures.get('extra_tokens')

            if strict:
                complete = False
                ## 12-05-2015
                if len(digits) == 3:
                    complete = True
                ## 19 February 2013 year 09:10
                elif (len(months) == 1) and (len(digits) == 2):
                    complete = True

                if not complete:
                    continue

            ## sanitize date string
            ## replace unhelpful whitespace characters with single whitespace
            match_str = re.sub('[\n\t\s\xa0]+', ' ', match_str)
            match_str = match_str.strip(self.STRIP_CHARS)

            ## Save sanitized source string
            yield match_str, indices, captures


def find_dates(text, source=False, index=False, strict=False):
    """
    Extract datetime strings from text
    :param text:
        A string that contains one or more natural language or literal
        datetime strings
    :type text: str|unicode
    :param source:
        Return the original string segment
    :type source: boolean
    :param index:
        Return the indices where the datetime string was located in text
    :type index: boolean
    :param strict:
        Only return datetimes with complete date information. For example:
        `July 2016` of `Monday` will not return datetimes.
        `May 16, 2015` will return datetimes.
    :type strict: boolean
    :return: Returns a generator that produces :mod:`datetime.datetime` objects,
        or a tuple with the source text and index, if requested
    """
    date_finder = DateFinder()
    matches = date_finder.find_dates(text, source=source, index=index, strict=strict)
    return [match for match in matches]

if __name__ == '__main__':
    match = find_dates(u"Hello a quelle heure ouvre le magasin rue francois 1er demain", source=True)
    for mt in match:
        print(mt)
    match = find_dates(u"Rendez vous lundi a 12h30", source=True)
    for mt in match:
        print(mt)