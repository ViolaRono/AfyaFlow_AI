# AfyaFlow
## Predicting outpatient demand at AIC Kijabe Hospital
AfyaFlow predicts how many patients will show up at each hospital department, hour by hour, so that staff and rooms can be put where the patients will be.
Moringa School, Capstone. Group 1, Queue Busters.

### The problem
Kijabe Hospital plans its outpatient departments around daily averages. But patients do not arrive in an even spread through the day. They come in a big wave in the morning.
We looked at 306,306 patient registrations. In the busiest single hour, one department received 59 patients. A typical department-hour gets about two. So a nurse rostered to the average is swamped at 9am and has nothing to do by 3pm.
AfyaFlow predicts arrivals ahead of time, so the roster can match the wave instead of the average.
#### What we found
We tested our model against a simple rule: assume a department will get the same number of patients it got at the same hour last week. 
That rule is close to what an experienced matron already guesses.
If our model could not beat it, it was not worth building.
Here is how it did. MAE means average error, in patients. Lower is better.
Forecast	Simple rule	Our model	Better by	Winner
Next hour	0.30	0.24	18%	LightGBM
Next day	3.05	2.48	19%	LightGBM
Next week	16.22	13.63	16%	LightGBM + Prophet together
Next month	25.98	40.83	it lost	the simple rule
The model wins on hourly and daily forecasts. Those are the ones used for rostering, so that is where it matters.

#### What the data told us about the hospital

* A few departments carry most of the load. 7 out of 44 departments account for about 70% of all visits: General OPD, Admission, MCH, Chronic Care Clinic, Speciality Clinic, Oncology and Renal.
* Mornings are the pressure point. Arrivals spike between 8am and 9am, and Mondays are the worst.
* Some patients are completely predictable. Renal Dialysis and Program patients come back more
than 99% of the time. These can simply be booked into fixed slots.
* Older patients return more often. Men and women return at the same rate. Return visits go up steadily with age, but gender makes almost no difference.
How we built it
```
Patient registrations
  → clean and remove duplicates
  → count patients per department per hour
  → fill in the quiet hours with zero
  → add calendar and recent-history columns
  → train on the older 80%, test on the newest 20%
  → compare against the simple rule
```
#### Three choices that matter:
* We tested on the newest data, not a random sample. If you pick test rows at random, the model
gets to see the future while it learns. 
* We split by date instead, training on the earlier period and testing on the later one.
We filled in the empty hours. The hospital system only records a row when someone actually
arrives. So we added the missing hours back with a count of zero. Without this the model would never
learn what a quiet afternoon looks like.
* We never let the model peek. Columns like "how many patients came last hour" are shifted back in
time before the model sees them, so it cannot accidentally read the answer it is trying to predict.

What the model looks at:

| Forecast | Columns used |
|---|---|
| Hourly | hour of day, day of week, weekend or not, month, last hour's count, average of last 3 hours, department |
| Daily | day of week, weekend or not, month, yesterday's count, same day last week, average of last 7 days, department |
| Weekly | month, quarter, year, last week, 2 weeks ago, 4 weeks ago, average of last 4 weeks, department |

## Why the hourly numbers look so small

An average error of 0.24 patients looks too good to be true. It is small because most department-hours have nobody in them at all, and those easy zeros pull the average down.

The fairer way to look at it is department by department:

| Department | Simple rule | Our model | Better by |
|---|---|---|---|
| General OPD | 2.35 | 1.76 | 25% |
| MEB Speciality Clinic | 0.86 | 0.61 | 29% |
| MCH | 1.08 | 0.80 | 26% |
| Admission | 1.40 | 1.08 | 23% |

The model earns its keep in the busy departments, not on the empty hours.

#### What we recommend to the hospital
1. Put more staff on weekday mornings, especially Mondays and the 8am to 9am peak.
2. Focus resources on the seven departments that handle about 70% of visits.
3. Use the hourly and daily forecasts for rostering. Treat the monthly ones as rough planning only.
4. Book Renal Dialysis and Program patients into fixed slots, since they almost always come back. This frees up space for walk-ins.
5. Add self-service check-in at registration to shorten the morning queue.
6. Set up a regular data refresh so the model keeps learning from new patients.

#### What is in this repo
```
AfyaFlow_AI/
├── notebooks/
│   └── AfyaFlow.ipynb          the full analysis
├── reports/
│   └── AfyaFlow_DataReport.docx
├── presentation/
│   └── AfyaFlow_Slides.pdf
└── README.md
```

*What this project cannot do*

* We were not given waiting times, staffing levels or bed availability. So we work out congestion
from how many people arrive, not from how long they actually waited.
* The data is a one-off export. For real use it would need to update automatically.
* This is a planning tool. It does not make any clinical decisions about patients.

#### Built with

Python, pandas, LightGBM, Prophet, scikit-learn, matplotlib, seaborn, Streamlit

#### *The team*

Silvia, Peter, Cate, Leon, Alisha, Viola
