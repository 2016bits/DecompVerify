import re


VAR_PATTERN = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")

NEGATION_CUE_PATTERNS = [
    r"\bnot\b",
    r"\bno\b",
    r"\bno longer\b",
    r"\bnever\b",
    r"\bwithout\b",
    r"\bfail(?:ed|s|ing)?\b",
    r"\bunsuccessful(?:ly)?\b",
    r"\bunable\b",
    r"\bunaware\b",
    r"\black(?:s|ed|ing)?\b",
    r"\babsence\b",
]

MONTH_TO_NUM = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

TEXT_NORMALIZATION_REPLACEMENTS = {
    "organisations": "organizations",
    "organisation": "organization",
    "programmes": "programs",
    "programme": "program",
    "millimetres": "millimeters",
    "millimetre": "millimeter",
    "metres": "meters",
    "metre": "meter",
    "kilometres": "kilometers",
    "kilometre": "kilometer",
    "centimetres": "centimeters",
    "centimetre": "centimeter",
    "colour": "color",
    "colours": "colors",
    "centre": "center",
    "centres": "centers",
}

TEMPORAL_NORMALIZATION_REPLACEMENTS = {
    "holocaust": "nazi german genocide against jews",
    "graduating college": "finish college",
    "graduated college": "finish college",
    "graduating university": "finish college",
    "graduated university": "finish college",
    "after finishing at university": "after finish college",
    "after finishing university": "after finish college",
    "after graduating college": "after finish college",
    "after graduating university": "after finish college",
    "after wartime service": "after war service",
    "in the wake of": "after",
    "wake of": "after",
    "months": "month",
    "years": "year",
    "weeks": "week",
    "days": "day",
    "infancy": "infant",
}

STOPWORD_TOKENS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "from", "by", "with",
}

GENERIC_LOCATION_TOKENS = {
    "country", "region", "area", "province", "state", "city", "town",
    "village", "county", "district", "place",
}

DIRECTION_TOKENS = {
    "north", "south", "east", "west",
    "northern", "southern", "eastern", "western",
    "northeast", "northwest", "southeast", "southwest",
    "northeastern", "northwestern", "southeastern", "southwestern",
}

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

SCALE_WORDS = {
    "hundred": 100,
    "thousand": 1000,
}

ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}


def _clean_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def apply_term_replacements(text, replacements):
    for src in sorted(replacements.keys(), key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(src)}\b", replacements[src], text)
    return text


def normalize_text_for_match(text):
    text = _clean_text(text).lower()
    text = apply_term_replacements(text, TEXT_NORMALIZATION_REPLACEMENTS)
    text = re.sub(r"(?<=\d)(?=[a-z%])", " ", text)
    text = re.sub(r"(?<=[a-z])(?=\d)", " ", text)
    text = re.sub(r"[^\w\s\.]", " ", text)
    text = text.replace(".", " ")
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_semantic_text(text):
    text = normalize_text_for_match(text)
    text = apply_term_replacements(text, TEMPORAL_NORMALIZATION_REPLACEMENTS)
    text = text.replace("all year round", "year round")
    text = text.replace("all year-round", "year round")
    text = text.replace("all-year round", "year round")
    text = text.replace("all-year-round", "year round")
    text = text.replace("year-round", "year round")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_explicit_negation(text):
    text_l = _clean_text(text).lower()
    if not text_l:
        return False
    return any(re.search(pattern, text_l) for pattern in NEGATION_CUE_PATTERNS)


def replace_placeholders(text, bindings):
    text = _clean_text(text)
    if not text or not isinstance(bindings, dict):
        return text
    for var in sorted(bindings.keys(), key=len, reverse=True):
        val = _clean_text(bindings.get(var, ""))
        if var.startswith("?") and val:
            text = text.replace(var, val)
    return text


def unresolved_vars(text):
    return sorted(set(VAR_PATTERN.findall(_clean_text(text))))


def normalize_status(status):
    status_l = _clean_text(status).lower()
    if status_l in {"support", "supported"}:
        return "supported"
    if status_l in {"contradict", "contradicted"}:
        return "contradicted"
    if status_l == "api_error":
        return "api_error"
    return "insufficient"


def normalize_yesno(answer):
    answer_l = _clean_text(answer).lower()
    if answer_l in {"yes", "yes.", "true"}:
        return "yes"
    if answer_l in {"no", "no.", "false"}:
        return "no"
    return None


def contains_negation(fact):
    return (fact.get("constraint", {}) or {}).get("negation") is True


def extract_years(text):
    return re.findall(r"\b([1-9][0-9]{2,3})\b", _clean_text(text))


def extract_dates(text):
    month_regex = "|".join(MONTH_TO_NUM.keys())
    text = _clean_text(text)
    patterns = [
        re.compile(rf"\b(?P<day>\d{{1,2}})\s+(?P<month>{month_regex})\s+(?P<year>\d{{3,4}})\b", re.IGNORECASE),
        re.compile(rf"\b(?P<month>{month_regex})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*|\s+)(?P<year>\d{{3,4}})\b", re.IGNORECASE),
    ]

    dates = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            month = match.group("month").lower()
            dates.append((int(match.group("year")), MONTH_TO_NUM[month], int(match.group("day"))))
    return dates


def extract_month_numbers(text):
    text_l = _clean_text(text).lower()
    return sorted({num for month, num in MONTH_TO_NUM.items() if re.search(rf"\b{month}\b", text_l)})


def is_year_only_text(text):
    text_l = _clean_text(text).lower()
    text_l = re.sub(r"\b[1-9][0-9]{2,3}\b", " ", text_l)
    text_l = re.sub(r"\b(ad|ce|bc|bce)\b", " ", text_l)
    text_l = re.sub(r"[^a-z]", " ", text_l)
    return not re.sub(r"\s+", "", text_l)


def extract_decade_range(text):
    text_l = _clean_text(text).lower()
    match = re.search(r"\b(?:(early|mid|late)\s+)?([1-9][0-9]{2,3})(?:'s|s)\b", text_l)
    if not match:
        return None

    modifier = match.group(1)
    start = int(match.group(2))
    span = 100 if start % 100 == 0 else 10
    end = start + span - 1

    if modifier == "early":
        end = start + (39 if span == 100 else 3)
    elif modifier == "mid":
        start = start + (30 if span == 100 else 3)
        end = start + (39 if span == 100 else 3)
    elif modifier == "late":
        start = start + (70 if span == 100 else 7)

    return start, end


def looks_temporal_text(text):
    norm = normalize_semantic_text(text)
    if not norm:
        return False
    if extract_dates(text) or extract_years(text) or extract_month_numbers(text):
        return True

    temporal_keywords = [
        "during", "after", "before", "until", "while", "when", "eventually", "initially",
        "first", "last", "spring", "summer", "fall", "autumn", "winter", "period", "era",
        "century", "month", "year", "week", "day", "decade", "later", "earlier", "life",
    ]
    return any(re.search(rf"\b{keyword}\b", norm) for keyword in temporal_keywords)


def temporal_phrase_match(answer, target):
    answer_tokens = set(tok for tok in normalize_semantic_text(answer).split() if tok not in STOPWORD_TOKENS)
    target_tokens = set(tok for tok in normalize_semantic_text(target).split() if tok not in STOPWORD_TOKENS)
    if not answer_tokens or not target_tokens:
        return False

    overlap = len(answer_tokens & target_tokens) / max(1, min(len(answer_tokens), len(target_tokens)))
    return overlap >= 0.75


def time_match(answer, times):
    ans = normalize_semantic_text(answer)
    targets = [_clean_text(value) for value in (times or []) if _clean_text(value)]
    if not targets:
        return None

    answer_dates = extract_dates(answer)
    answer_months = extract_month_numbers(answer)
    answer_years = [int(value) for value in extract_years(answer)]
    answer_year_span = (min(answer_years), max(answer_years)) if answer_years else None
    saw_conflict = False

    for target in targets:
        target_norm = normalize_semantic_text(target)
        if target_norm and (target_norm in ans or ans in target_norm):
            return True

        if target_norm and temporal_phrase_match(ans, target_norm):
            return True

        target_dates = extract_dates(target)
        if target_dates:
            if answer_dates:
                if any(answer_date == target_date for answer_date in answer_dates for target_date in target_dates):
                    return True
                saw_conflict = True
            continue

        if "second half of the year" in target_norm or "second half" in target_norm:
            if any(month >= 7 for month in answer_months):
                return True
            if answer_months:
                saw_conflict = True
            continue

        if "first half of the year" in target_norm or "first half" in target_norm:
            if any(month <= 6 for month in answer_months):
                return True
            if answer_months:
                saw_conflict = True
            continue

        target_months = extract_month_numbers(target)
        target_years = [int(value) for value in extract_years(target)]

        if target_months and target_years:
            if answer_months and answer_years:
                if any(month in target_months for month in answer_months) and any(year in target_years for year in answer_years):
                    return True
                saw_conflict = True
            continue

        decade_range = extract_decade_range(target)
        if decade_range and answer_years:
            start, end = decade_range
            if any(start <= year <= end for year in answer_years):
                return True
            saw_conflict = True
            continue

        if target_years and answer_years:
            if re.search(r"\b(around|about|circa|approximately|roughly)\b", target_norm):
                target_year = target_years[0]
                if answer_year_span and answer_year_span[0] <= target_year <= answer_year_span[1]:
                    return True
                saw_conflict = True
                continue

            if is_year_only_text(target):
                if any(year in target_years for year in answer_years):
                    return True
                saw_conflict = True
                continue

            if set(answer_years) == set(target_years):
                return True
            saw_conflict = True

    return False if saw_conflict else None


def normalize_num_string(text):
    text = _clean_text(text).lower()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", text)
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    text = re.sub(r"(?<=\d)(?=[a-z%])", " ", text)
    text = re.sub(r"(?<=[a-z])(?=\d)", " ", text)
    text = text.replace("-", " ")
    return text.strip()


def tokenize_number_words(text):
    text = normalize_num_string(text)
    text = re.sub(r"[^\w\s]", " ", text)
    return [tok for tok in text.split() if tok]


def consume_number_tokens(tokens, start):
    total = 0
    current = 0
    used = 0
    found = False

    while start + used < len(tokens):
        tok = tokens[start + used]
        if tok in NUMBER_WORDS:
            current += NUMBER_WORDS[tok]
            found = True
        elif tok in SCALE_WORDS:
            current = max(1, current) * SCALE_WORDS[tok]
            if SCALE_WORDS[tok] >= 1000:
                total += current
                current = 0
            found = True
        elif tok in ORDINAL_WORDS:
            current += ORDINAL_WORDS[tok]
            found = True
            used += 1
            break
        elif tok == "and" and found:
            used += 1
            continue
        else:
            break
        used += 1

    if not found:
        return None, 0
    return total + current, used


def extract_numeric_values(text):
    text = normalize_num_string(text)
    values = []

    for match in re.finditer(r"(?<!\w)(\d+(?:\.\d+)?)(?:st|nd|rd|th)?(?=(?:\s|$|[a-z%]))", text.lower()):
        num = float(match.group(1))
        values.append(int(num) if num.is_integer() else num)

    tokens = tokenize_number_words(text)
    idx = 0
    while idx < len(tokens):
        value, used = consume_number_tokens(tokens, idx)
        if used:
            values.append(value)
            idx += used
        else:
            idx += 1

    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


UNIT_ALIASES = {
    "ha": "area",
    "hectare": "area",
    "hectares": "area",
    "acre": "area",
    "acres": "area",
    "sq": "area",
    "square": "area",
}


def extract_numeric_values_with_units(text):
    text = normalize_num_string(text)
    unit_regex = "|".join(sorted((re.escape(unit) for unit in UNIT_ALIASES), key=len, reverse=True))
    values = []
    for match in re.finditer(rf"(?<!\w)(\d+(?:\.\d+)?)(?:st|nd|rd|th)?\s+(?P<unit>{unit_regex})\b", text.lower()):
        num = float(match.group(1))
        value = int(num) if num.is_integer() else num
        values.append((value, UNIT_ALIASES[match.group("unit")]))
    return values


def select_unit_compatible_numbers(answer, target):
    target_units = {unit for _, unit in extract_numeric_values_with_units(target)}
    if not target_units:
        return extract_numeric_values(answer)

    unit_numbers = [
        value
        for value, unit in extract_numeric_values_with_units(answer)
        if unit in target_units
    ]
    return unit_numbers or extract_numeric_values(answer)


def under_over_is_nonquantitative(text, match, cue):
    if cue not in {"under", "over"}:
        return False

    before = text[:match.start()].strip().split()
    previous = before[-1] if before else ""
    if cue == "under" and previous in {"bring", "brings", "brought", "put", "puts", "placed", "place", "placing"}:
        return True

    tail = text[match.end():]
    next_match = re.match(r"\s+([a-z]+)", tail)
    next_word = next_match.group(1) if next_match else ""
    if cue == "under" and next_word in {
        "general", "generals", "government", "governments", "administration",
        "command", "control", "leadership", "system", "systems", "rule", "rules",
        "authority", "authorities", "banner", "name", "title",
    }:
        return True

    return False


def parse_quantity_comparator(text):
    text = normalize_num_string(text)
    number_token = r"(?:\d+(?:\.\d+)?(?:st|nd|rd|th)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth)"
    number_phrase = rf"(?P<num>{number_token}(?:\s+(?:and\s+)?{number_token}){{0,4}})"
    patterns = [
        (rf"\b(?P<cue>more than|over|greater than|above)\s+{number_phrase}\b", ">"),
        (rf"\b(?P<cue>at least|no fewer than|minimum of)\s+{number_phrase}\b", ">="),
        (rf"\b(?P<cue>at most|no more than|up to|maximum of)\s+{number_phrase}\b", "<="),
        (rf"\b(?P<cue>less than|under|below|fewer than)\s+{number_phrase}\b", "<"),
    ]
    for pattern, op in patterns:
        match = re.search(pattern, text)
        if match:
            if under_over_is_nonquantitative(text, match, match.group("cue")):
                continue
            nums = extract_numeric_values(match.group("num"))
            if nums:
                return op, nums[0]
    return None


def approximate_quantity_match(quantity, answer_numbers):
    quantity_norm = normalize_num_string(quantity)
    quantity_numbers = extract_numeric_values(quantity_norm)
    if not quantity_numbers or not answer_numbers:
        return None

    if re.search(r"\b(?:story|plot|film|movie|series|book|novel|episode|documentary|article|song)\b.{0,50}\babout\s+\w+", quantity_norm):
        return None

    target = float(quantity_numbers[0])
    tolerance = 1.0 if abs(target) <= 20 else max(1.0, abs(target) * 0.05)

    if re.search(r"\b(about|around|approximately|roughly)\b", quantity_norm):
        return any(abs(float(num) - target) <= tolerance for num in answer_numbers)

    if re.search(r"\b(almost|nearly)\b", quantity_norm):
        return any((target - tolerance) <= float(num) < target for num in answer_numbers)

    return None


def fractional_quantity_match(quantity, answer_numbers, quantities, fact_text):
    quantity_norm = normalize_num_string(quantity)
    if "half" not in quantity_norm or not answer_numbers:
        return None

    context_numbers = []
    for other in quantities:
        if other == quantity:
            continue
        context_numbers.extend(num for num in extract_numeric_values(other) if float(num) > 1)

    if not context_numbers:
        context_numbers.extend(num for num in extract_numeric_values(fact_text) if float(num) > 1)

    if not context_numbers:
        return None

    half = float(context_numbers[0]) / 2.0
    answer_values = [float(num) for num in answer_numbers]

    if re.search(r"\b(over|more than) half\b", quantity_norm):
        return any(num > half for num in answer_values)
    if re.search(r"\b(under|less than) half\b", quantity_norm):
        return any(num < half for num in answer_values)
    if re.search(r"\bhalf\b", quantity_norm):
        return any(abs(num - half) <= 0.5 for num in answer_values)

    return None


def compare_numeric(value, op, target):
    if op == ">":
        return value > target
    if op == ">=":
        return value >= target
    if op == "<":
        return value < target
    if op == "<=":
        return value <= target
    return value == target


def infer_quantity_targets_from_fact(fact_text):
    fact_text = _clean_text(fact_text)
    fact_l = fact_text.lower()
    if re.search(
        r"\b(multiple|several|many|numerous|various|different|majority|more popular|less popular|grew|grown|growing|increase|increased|decrease|decreased|over|under|at least|at most|less than|more than|fewer than|about|around|approximately|roughly|almost|nearly|half|once|twice|thrice|often|common|frequent|frequently|rare|broad|wide|narrow|infant|infancy|short time|short period|short periods|month|months|year|years|week|weeks|day|days)\b",
        fact_l,
    ):
        return [fact_text]
    if re.search(r"\d+(?:\.\d+)?(?:st|nd|rd|th)?(?=[a-z%]*\b)", fact_l):
        return [fact_text]
    if any(word in fact_l for word in list(NUMBER_WORDS.keys()) + list(ORDINAL_WORDS.keys())):
        return [fact_text]
    return []


def qualitative_quantity_match(fact_norm, answer_norm, answer_numbers):
    plural_synonyms = {"multiple", "several", "many", "numerous", "various", "different"}

    if "multiple" in fact_norm:
        if any(float(num) >= 2 for num in answer_numbers) or any(word in answer_norm for word in plural_synonyms):
            return True
        if any(float(num) == 1 for num in answer_numbers):
            return False

    if "several" in fact_norm:
        if any(float(num) >= 3 for num in answer_numbers) or any(word in answer_norm for word in {"several", "many", "numerous"}):
            return True
        if answer_numbers and all(float(num) < 3 for num in answer_numbers):
            return False

    if re.search(r"\b(many|numerous)\b", fact_norm):
        if any(float(num) >= 3 for num in answer_numbers) or any(word in answer_norm for word in {"many", "numerous", "several"}):
            return True

    if re.search(r"\b(various|different)\b", fact_norm):
        if any(float(num) >= 2 for num in answer_numbers) or any(word in answer_norm for word in {"various", "different", "several", "multiple"}):
            return True

    if "majority" in fact_norm:
        if "majority" in answer_norm:
            return True
        if "minority" in answer_norm:
            return False

    if re.search(r"\b(once|single)\b", fact_norm):
        if "once" in answer_norm or any(float(num) == 1 for num in answer_numbers):
            return True
        if answer_numbers and all(float(num) != 1 for num in answer_numbers):
            return False

    if re.search(r"\b(twice|two time|two times|second time)\b", fact_norm):
        if "twice" in answer_norm or any(float(num) == 2 for num in answer_numbers):
            return True
        if answer_numbers and all(float(num) != 2 for num in answer_numbers):
            return False

    if re.search(r"\b(thrice|three time|three times|third time)\b", fact_norm):
        if "thrice" in answer_norm or any(float(num) == 3 for num in answer_numbers):
            return True
        if answer_numbers and all(float(num) != 3 for num in answer_numbers):
            return False

    if re.search(r"\b(often|common|frequent|frequently)\b", fact_norm):
        if re.search(r"\b(often|common|frequent|frequently|regularly)\b", answer_norm):
            return True
        if re.search(r"\b(rare|rarely|uncommon)\b", answer_norm):
            return False

    if re.search(r"\b(rare|rarely)\b", fact_norm):
        if re.search(r"\b(rare|rarely|uncommon)\b", answer_norm):
            return True
        if re.search(r"\b(common|often|frequent|frequently)\b", answer_norm):
            return False

    if re.search(r"\binfant\b", fact_norm):
        if re.search(r"\b(infant|infancy)\b", answer_norm):
            return True

    if re.search(r"\bbroad\b", fact_norm):
        if re.search(r"\b(broad|wide)\b", answer_norm):
            return True
        if re.search(r"\bnarrow\b", answer_norm):
            return False

    if re.search(r"\bnarrow\b", fact_norm):
        if re.search(r"\bnarrow\b", answer_norm):
            return True
        if re.search(r"\b(broad|wide)\b", answer_norm):
            return False

    if re.search(r"\bshort (?:time|period)\b", fact_norm):
        if re.search(r"\bshort\b", answer_norm):
            return True

    if re.search(r"\b(more popular|grew|grown|growing|increase|increased|higher|greater)\b", fact_norm):
        if re.search(r"\b(grow|grew|grown|growing|increase|increased|increasing|rose|risen|higher|greater|more)\b", answer_norm):
            return True
        if re.search(r"\b(decrease|decreased|decline|declined|less|lower)\b", answer_norm):
            return False

    if re.search(r"\b(less popular|decrease|decreased|lower|fewer|smaller)\b", fact_norm):
        if re.search(r"\b(decrease|decreased|decline|declined|less|lower|fewer|smaller)\b", answer_norm):
            return True
        if re.search(r"\b(grow|grew|grown|increase|increased|more|higher|greater)\b", answer_norm):
            return False

    return None


def quantity_match(answer, quantities, fact_text=""):
    answer_norm = normalize_num_string(answer)
    quantity_values = [normalize_num_string(value) for value in (quantities or []) if _clean_text(value)]
    quantity_values.extend(normalize_num_string(value) for value in infer_quantity_targets_from_fact(fact_text))

    if not quantity_values:
        return None

    deduped = []
    for value in quantity_values:
        if value and value not in deduped:
            deduped.append(value)
    quantity_values = deduped

    answer_numbers = extract_numeric_values(answer)
    answer_comparator = parse_quantity_comparator(answer)
    saw_conflict = False

    for quantity in quantity_values:
        compatible_answer_numbers = select_unit_compatible_numbers(answer, quantity)
        comparator = parse_quantity_comparator(quantity)
        if comparator:
            if answer_comparator == comparator:
                return True
            if compatible_answer_numbers:
                op, target = comparator
                if any(compare_numeric(float(num), op, float(target)) for num in compatible_answer_numbers):
                    return True
                saw_conflict = True
                continue

        approximate = approximate_quantity_match(quantity, compatible_answer_numbers)
        if approximate is True:
            return True
        if approximate is False:
            saw_conflict = True
            continue

        fractional = fractional_quantity_match(quantity, compatible_answer_numbers, quantity_values, fact_text)
        if fractional is True:
            return True
        if fractional is False:
            saw_conflict = True
            continue

        quantity_numbers = extract_numeric_values(quantity)
        if quantity_numbers and compatible_answer_numbers:
            if any(abs(float(answer_num) - float(quantity_num)) <= 0.05 for answer_num in compatible_answer_numbers for quantity_num in quantity_numbers):
                return True
            saw_conflict = True
            continue

        if answer_norm and (answer_norm in quantity or quantity in answer_norm):
            return True

        qualitative = qualitative_quantity_match(quantity, answer_norm, answer_numbers)
        if qualitative is True:
            return True
        if qualitative is False:
            saw_conflict = True

    return False if saw_conflict else None


def span_grounded_in_evidence(evidence_span, gold_evidence):
    span = normalize_text_for_match(evidence_span)
    gold = normalize_text_for_match(gold_evidence)
    if not span:
        return False

    if span in gold:
        return True

    span_tokens = set(span.split())
    gold_tokens = set(gold.split())
    if not span_tokens:
        return False

    overlap = len(span_tokens & gold_tokens) / max(1, len(span_tokens))
    return overlap >= 0.7


def content_tokens(text):
    return [tok for tok in normalize_text_for_match(text).split() if tok and tok not in STOPWORD_TOKENS]


def directional_location_match(a, b):
    a_tokens = set(content_tokens(a))
    b_tokens = set(content_tokens(b))
    if not a_tokens or not b_tokens:
        return False

    shared_directions = (a_tokens & b_tokens) & DIRECTION_TOKENS
    if not shared_directions:
        return False

    a_core = a_tokens - GENERIC_LOCATION_TOKENS
    b_core = b_tokens - GENERIC_LOCATION_TOKENS
    if a_core <= shared_directions or b_core <= shared_directions:
        return True

    if "country" in a_tokens or "country" in b_tokens:
        return True

    return False


def text_match_loose(a, b):
    a_norm = normalize_text_for_match(a)
    b_norm = normalize_text_for_match(b)
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm or a_norm in b_norm or b_norm in a_norm:
        return True
    if directional_location_match(a, b):
        return True

    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    if not a_tokens or not b_tokens:
        return False

    overlap = len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))
    return overlap >= 0.75


def extract_entity_target_from_fact(fact_text):
    text = _clean_text(fact_text)

    strict_patterns = [
        r"\bwas born in\s+(.+?)[\.\,]?$",
        r"\bshown on\s+(.+?)[\.\,]?$",
        r"\blocated in the\s+(.+?)[\.\,]?$",
        r"\blocated in\s+(.+?)[\.\,]?$",
        r"\bwas located in\s+(.+?)[\.\,]?$",
        r"\btook place in\s+(.+?)[\.\,]?$",
        r"\bcan only be used in\s+(.+?)[\.\,]?$",
        r"\bknown by\s+(.+?)[\.\,]?$",
        r"\bis in\s+([A-Z][A-Za-z0-9 ,\-/]+)[\.\,]?$",
    ]
    for pattern in strict_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            candidate = _clean_text(match.group(1))
            if candidate and candidate.lower() not in {"a village", "a town", "a city", "a television sitcom", "a film", "an animal"}:
                return candidate
    return ""


def is_existential_or_type_fact(fact_text):
    text = _clean_text(fact_text).lower()
    patterns = [
        "there exists",
        "is a village",
        "is a town",
        "is a city",
        "is a character in a television sitcom",
        "is a character in",
        "was born in a village",
        "provided the score for a film",
        "played for a team",
        "has a star",
    ]
    return any(pattern in text for pattern in patterns)


def verify_entity_wh(fact, answer, answer_status, evidence_span, gold_evidence):
    fact_text = _clean_text(fact.get("text", ""))
    grounded = span_grounded_in_evidence(evidence_span, gold_evidence)
    target = extract_entity_target_from_fact(fact_text)

    answer_text = _clean_text(answer)
    answer_missing = answer_status in {"insufficient", "api_error"} or answer_text.lower() == "insufficient" or not answer_text
    candidate = answer_text

    if answer_missing and grounded and target and text_match_loose(evidence_span, target):
        candidate = _clean_text(evidence_span)
        answer_missing = False

    if answer_missing:
        return "insufficient", "The answer does not provide enough information."

    if is_existential_or_type_fact(fact_text) or not target:
        if answer_status == "contradicted":
            return "contradict", "The answer result contradicts the fact."
        if grounded and candidate:
            return "support", "The answer provides a grounded value for the fact."
        return "insufficient", "The answer is not sufficiently grounded in the evidence."

    if text_match_loose(candidate, target):
        if grounded or span_grounded_in_evidence(candidate, gold_evidence):
            return "support", "The answer matches the target value and is grounded in the evidence."
        return "insufficient", "The answer matches the target value but is not grounded enough."

    return "contradict", "The answer conflicts with the target value in the atomic fact."
