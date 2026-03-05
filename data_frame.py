import pandas as pd

df = pd.read_csv("workout_fitness_tracker_data.csv")
df.columns = [c.strip() for c in df.columns]

df = df[['User ID', 'Workout Type', 'Workout Duration (mins)',
         'Calories Burned', 'Workout Intensity']]

def get_duration_bucket(duration):
    if duration <= 30:
        return '0-30'
    elif duration <= 60:
        return '31-60'
    elif duration <= 90:
        return '61-90'
    else:
        return '91+'

df['Duration Bucket'] = df['Workout Duration (mins)'].apply(get_duration_bucket)

workouts_df = (
    df.drop_duplicates(subset=['Workout Type', 'Workout Duration (mins)',
                               'Calories Burned', 'Workout Intensity'])
      .reset_index(drop=True)
)

workouts_df['Workout ID'] = workouts_df.index

#user logd
user_logs = df.merge(
    workouts_df[['Workout Type', 'Workout Intensity',
                 'Workout Duration (mins)', 'Workout ID']],
    on=['Workout Type', 'Workout Duration (mins)', 'Workout Intensity'],
    how='left'
)

user_logs = user_logs.sort_values('User ID').reset_index(drop=True)
user_logs['Session Index'] = user_logs.groupby('User ID').cumcount()

print(user_logs.tail())
print(workouts_df.tail())
