# NSW calendar sources for Task 11

`school_terms.csv` contains inclusive **administrative term boundaries** for NSW
government schools, Eastern Division, 2010-2021. Staff development days remain
inside terms. It is a regional calendar proxy, not a claim about the attendance
of every NSW student. Private schools, Western Division and individual school
variations are not represented. Bankstown is in the Eastern Division.

`is_school_holiday = 1` outside these intervals, including weekends adjoining a
vacation and the January/December portions of summer holidays. Ordinary weekends
inside a term have `is_school_holiday = 0` and `is_weekend = 1`. This intentionally
differs from official vacation lists that sometimes list weekdays only.
Unscheduled closures/remote learning in 2020 are NOT encoded as advance-known
school holidays. Calendar publication vintages have not been reconstructed:
scheduled dates are assumed available in advance; this is a backtest limitation.

Replacement sources checked 10 September 2026. The original 2010-2015 school
PDF links no longer work. The table below provides replacement evidence; the
existing `source_id` values are retained so the CSV does not need to change.
These IDs name the original references, not the replacement publishers.

| Existing source_id | Coverage | Replacement source and check |
|---|---|---|
| newcastle_2009 | 2010 | [University of Sydney International Student Guide 2010, hosted by the University of Wisconsin](https://my.studyabroad.wisc.edu/File/Download/CB9549D3-7E79-4BAD-82A3-62CD7C56E751), PDF page 54. All four term ranges match the CSV. |
| glenwood_2011 | 2011 | [Parramatta Council archive: NSW Government Schools Term Dates](https://businesspapers.parracity.nsw.gov.au/Open/2011/06/OC_27062011_AGN_ATTACHMENT_2280_4.HTM). All four term ranges match the CSV. Search the page for ?NSW Government Schools Term Dates?. |
| coleambally_2011 | 2012 | [Parramatta Council archive: NSW Government Schools Term Dates](https://businesspapers.parracity.nsw.gov.au/Open/2011/06/OC_27062011_AGN_ATTACHMENT_2280_4.HTM). All four term ranges match the CSV. |
| castlehill_2012 | 2013 | Original reference: Castle Hill High, 2 November 2012, page 7 (link unavailable). A [third-party reproduction of an Education Gazette calendar](https://doczz.net/doc/1176532/--seaforth-public-school) corroborates all four term ranges, but access is inconsistent. A dependable original-source link is still needed. |
| tuggerawong_2014 | 2014 | [Transport for NSW school road-safety guide, hosted by NRSPP](https://media.nrspp.org.au/wp-content/uploads/2017/07/06021420/ROAD-SAFETY-GUIDE-FOR-SCHOOL-COMMUNITIES.pdf), PDF page 7. All four term ranges match the CSV. |
| campbelltown_2015 | 2015 | [Transport for NSW school road-safety guide, hosted by NRSPP](https://media.nrspp.org.au/wp-content/uploads/2017/07/06021420/ROAD-SAFETY-GUIDE-FOR-SCHOOL-COMMUNITIES.pdf), PDF page 7. All four term ranges match the CSV. |
| nsw_education | 2016-2021 | [NSW Department of Education, future and past term dates](https://education.nsw.gov.au/schooling/calendars/future-and-past-nsw-term-and-vacation-dates). Original reference retained. |

The replacement sources for 2010-2012 and 2014-2015 were retrieved directly,
and Arman confirmed those links open in his browser. No term dates were changed.
The 2013 reference remains a verification limitation, rather than a confirmed
working original source.

An older RTA calendar and the 2010 university guide list 26 April as the 2011
Term 2 start. The council's 2011 table lists 27 April, matching our CSV; it also
identifies 26 April as the substituted Easter Monday public holiday. Use the
council table for 2011, not the university guide's advance 2011 calendar.

Public holidays are generated offline using **holidays 0.104**, country `AU`,
subdivision `NSW`, observed days enabled, default PUBLIC category, with the
following explicit historical corrections in `public_holiday_calendar()`:

- Exclude the bank-only day (the package includes it for 2010).
- Remove Sunday 25 April and Sunday 26 December 2010; the old rules substituted
  Monday for the Sunday, rather than declaring both days public holidays.
- Add 28 December 2010, declared by a special proclamation.
- Add 26 April 2011 as substituted Easter Monday, retaining 25 April as ANZAC Day.

These corrections are backed by the [historical Banks and Bank Holidays Act](https://legislation.nsw.gov.au/view/whole/html/repealed/current/act-1912-043),
the [February 2010 government gazette proclamation](https://gazette.nsw.gov.au/gazette/2010/2/2010-31.pdf),
and [Schedule 1 clause 3 of the Public Holidays Act](https://legislation.nsw.gov.au/view/whole/html/2011-01-01/act-2010-115).
The [2010 parliamentary explanation](https://www.parliament.nsw.gov.au/bill/files/1729/LC%20114%20and%2011510.pdf)
also enumerates the 2010/11 Christmas/New Year dates. Local show holidays are excluded.

- [Package documentation and its NSW legislative references](https://holidays.readthedocs.io/en/latest/auto_gen_docs/australia/)
- [NSW Government public-holiday reference](https://www.nsw.gov.au/about-nsw/public-holidays)

`public_holidays.csv` is the generated review copy for 2010-2021, including these
corrections. Regenerate it with `public_holiday_calendar(range(2010, 2022))`;
runtime features use that function, not an independently maintained CSV.
`split_manifest.csv` is similarly generated from `default_splits()`. Tests
cover Easter, Australia Day substitution, Christmas substitutions, the 2011
Easter/ANZAC overlap and the exclusion of bank-only holidays. Sources and package
version are recorded for reproducibility; the public-holiday list is not
represented as a manually checked government extract for every date.
