from dateutil import parser

date_str1 = "2023-10-27 10:30:00"
date_str2 = "Oct 27, 2023 10:30 AM"
date_str3 = "27/11/2024 11:30"

dates = [date_str1, date_str2, date_str3]  # Store dates in a list

parsed_dates = [parser.parse(date) for date in dates] # Parse all dates

min_date = min(parsed_dates)
max_date = max(parsed_dates)
date_span = max_date - min_date
year,days=divmod(date_span.days,365)
hours, remainder = divmod(date_span.seconds, 3600)
months,days=divmod(days,30)
minutes, seconds = divmod(remainder, 60)
data_span=""
if year:
    data_span+=f"{year} year"
if months:
    data_span+=f", {months} months"
if days:
    data_span+=f", {days} days"
if minutes:
    data_span+=f", {minutes} minutes, "
if seconds:
    data_span+=f", {seconds} seconds"
print(data_span)
