# train_events = pd.read_csv("C:/Users/joyhc/OneDrive/Desktop/Kaggle Project/child-mind-institute-detect-sleep-states/train_events.csv")

os.chdir('C:\\Users\\joyhc\\OneDrive\\Desktop\\Kaggle Project\\child-mind-institute-detect-sleep-states')
import gc
# from copy import deepcopy
# from functools import partial
# from itertools import combinations
# from itertools import groupby
# from tqdm import tqdm
import polars as pl
import datetime
import pandas as pd

#
# class PATHS:
#     '''
#     Belongs in a separate file...
#     '''
#
#     MAIN_DIR = "C:\\Users\\joyhc\\OneDrive\\Desktop\\Kaggle Project\\child-mind-institute-detect-sleep-states\\"
#     # CSV FILES :
#     SUBMISSION = MAIN_DIR + "sample_submission.csv"
#     TRAIN_EVENTS = MAIN_DIR + "train_events.csv"
#     # PARQUET FILES:
#     TRAIN_SERIES = MAIN_DIR + "train_series.parquet"
#     TEST_SERIES = MAIN_DIR + "test_series.parquet"
#
#     @classmethod
#     def chdir(cls, main_dir='C:\\Users\\joyhc\\OneDrive\\Desktop\\Kaggle Project\\child-mind-institute-detect-sleep-states\\'):
#         if main_dir: cls.MAIN_DIR = main_dir

# Importing data

# Column transformations

dt_transforms = [
    pl.col('timestamp').str.to_datetime(),
    (pl.col('timestamp').str.to_datetime().dt.year()-2000).cast(pl.UInt8).alias('year'),
    pl.col('timestamp').str.to_datetime().dt.month().cast(pl.UInt8).alias('month'),
    pl.col('timestamp').str.to_datetime().dt.day().cast(pl.UInt8).alias('day'),
    pl.col('timestamp').str.to_datetime().dt.hour().cast(pl.UInt8).alias('hour')
]

data_transforms = [
    pl.col('anglez').cast(pl.Int16), # Casting anglez to 16 bit integer
    (pl.col('enmo')*1000).cast(pl.UInt16), # Convert enmo to 16 bit uint
]

train_series = pl.scan_parquet('/Users/joyhc/OneDrive/Desktop/Kaggle Project/child-mind-institute-detect-sleep-states/train_series.parquet').with_columns(
    dt_transforms + data_transforms
    )

train_events = pl.read_csv('/Users/joyhc/OneDrive/Desktop/Kaggle Project/child-mind-institute-detect-sleep-states/train_events.csv').with_columns(
    dt_transforms
    )

test_series = pl.scan_parquet('/Users/joyhc/OneDrive/Desktop/Kaggle Project/child-mind-institute-detect-sleep-states/test_series.parquet').with_columns(
    dt_transforms + data_transforms
    )

# Getting series ids as a list for convenience
series_ids = train_events['series_id'].unique(maintain_order=True).to_list()

# Removing series with mismatched counts:
onset_counts = train_events.filter(pl.col('event')=='onset').group_by('series_id').count().sort('series_id')['count']
wakeup_counts = train_events.filter(pl.col('event')=='wakeup').group_by('series_id').count().sort('series_id')['count']

counts = pl.DataFrame({'series_id':sorted(series_ids), 'onset_counts':onset_counts, 'wakeup_counts':wakeup_counts})
count_mismatches = counts.filter(counts['onset_counts'] != counts['wakeup_counts'])

train_series = train_series.filter(~pl.col('series_id').is_in(count_mismatches['series_id']))
train_events = train_events.filter(~pl.col('series_id').is_in(count_mismatches['series_id']))

# Updating list of series ids, not including series with no non-null values.
series_ids = train_events.drop_nulls()['series_id'].unique(maintain_order=True).to_list()

features, feature_cols = [pl.col('hour')], ['hour']

for mins in [5, 30, 60*2, 60*8]:
    features += [
        pl.col('enmo').rolling_mean(12 * mins, center=True, min_periods=1).abs().cast(pl.UInt16).alias(f'enmo_{mins}m_mean'),
        pl.col('enmo').rolling_max(12 * mins, center=True, min_periods=1).abs().cast(pl.UInt16).alias(f'enmo_{mins}m_max')
    ]

    feature_cols += [
        f'enmo_{mins}m_mean', f'enmo_{mins}m_max'
    ]

    # Getting first variations
    for var in ['enmo', 'anglez'] :
        features += [
            (pl.col(var).diff().abs().rolling_mean(12 * mins, center=True, min_periods=1)*10).abs().cast(pl.UInt32).alias(f'{var}_1v_{mins}m_mean'),
            (pl.col(var).diff().abs().rolling_max(12 * mins, center=True, min_periods=1)*10).abs().cast(pl.UInt32).alias(f'{var}_1v_{mins}m_max')
        ]

        feature_cols += [
            f'{var}_1v_{mins}m_mean', f'{var}_1v_{mins}m_max'
        ]

id_cols = ['series_id', 'step', 'timestamp']
##subseries_ids = test_series.clone().select('series_id').unique().collect()['series_id'].to_list()
##train_data = train_series.clone().filter(pl.col('series_id').is_in(subseries_ids)).with_columns(features).collect(streaming = True)

train_series = train_series.with_columns(
    features
).select(id_cols + feature_cols)

test_series = test_series.with_columns(
    features
).select(id_cols + feature_cols)


def make_train_dataset(train_data, train_events, drop_nulls=False):
    series_ids = train_data['series_id'].unique(maintain_order=True).to_list()
    X, y = pl.DataFrame(), pl.DataFrame()
    for idx in tqdm(series_ids):

        # Normalizing sample features
        sample = train_data.filter(pl.col('series_id') == idx).with_columns(
            [(pl.col(col) / pl.col(col).std()).cast(pl.Float32) for col in feature_cols if col != 'hour']
        )

        events = train_events.filter(pl.col('series_id') == idx)

        if drop_nulls:
            # Removing datapoints on dates where no data was recorded
            sample = sample.filter(
                pl.col('timestamp').dt.date().is_in(events['timestamp'].dt.date())
            )

        X = X.vstack(sample[id_cols + feature_cols])

        onsets = events.filter((pl.col('event') == 'onset') & (pl.col('step') != None))['step'].to_list()
        wakeups = events.filter((pl.col('event') == 'wakeup') & (pl.col('step') != None))['step'].to_list()

        # NOTE: This will break if there are event series without any recorded onsets or wakeups
        y = y.vstack(sample.with_columns(
            sum([(onset <= pl.col('step')) & (pl.col('step') <= wakeup) for onset, wakeup in
                 zip(onsets, wakeups)]).cast(pl.Boolean).alias('asleep')
        ).select('asleep')
                     )

    y = y.to_numpy().ravel()

    return X, y


def get_events(series, classifier):
    '''
    Takes a time series and a classifier and returns a formatted submission dataframe.
    '''

    series_ids = series['series_id'].unique(maintain_order=True).to_list()
    events = pl.DataFrame(schema={'series_id': str, 'step': int, 'event': str, 'score': float})

    for idx in tqdm(series_ids):

        # Collecting sample and normalizing features
        scale_cols = [col for col in feature_cols if (col != 'hour') & (series[col].std() != 0)]
        X = series.filter(pl.col('series_id') == idx).select(id_cols + feature_cols).with_columns(
            [(pl.col(col) / series[col].std()).cast(pl.Float32) for col in scale_cols]
        )

        # Applying classifier to get predictions and scores
        preds, probs = classifier.predict(X[feature_cols]), classifier.predict_proba(X[feature_cols])[:, 1]

        # NOTE: Considered using rolling max to get sleep periods excluding <30 min interruptions, but ended up decreasing performance
        X = X.with_columns(
            pl.lit(preds).cast(pl.Int8).alias('prediction'),
            pl.lit(probs).alias('probability')
        )

        # Getting predicted onset and wakeup time steps
        pred_onsets = X.filter(X['prediction'].diff() > 0)['step'].to_list()
        pred_wakeups = X.filter(X['prediction'].diff() < 0)['step'].to_list()

        if len(pred_onsets) > 0:

            # Ensuring all predicted sleep periods begin and end
            if min(pred_wakeups) < min(pred_onsets):
                pred_wakeups = pred_wakeups[1:]

            if max(pred_onsets) > max(pred_wakeups):
                pred_onsets = pred_onsets[:-1]

            # Keeping sleep periods longer than 30 minutes
            sleep_periods = [(onset, wakeup) for onset, wakeup in zip(pred_onsets, pred_wakeups) if
                             wakeup - onset >= 12 * 30]

            for onset, wakeup in sleep_periods:
                # Scoring using mean probability over period
                score = X.filter((pl.col('step') >= onset) & (pl.col('step') <= wakeup))['probability'].mean()

                # Adding sleep event to dataframe
                events = events.vstack(pl.DataFrame().with_columns(
                    pl.Series([idx, idx]).alias('series_id'),
                    pl.Series([onset, wakeup]).alias('step'),
                    pl.Series(['onset', 'wakeup']).alias('event'),
                    pl.Series([score, score]).alias('score')
                ))

    # Adding row id column
    events = events.to_pandas().reset_index().rename(columns={'index': 'row_id'})

    return events

# We will collect datapoints and take 1 million samples

train_data = train_series.filter(pl.col('series_id').is_in(series_ids)).collect().sample(int(1e6))

