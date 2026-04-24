import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from icalendar import Calendar
from datetime import datetime, time
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="WFH Burnout & Recovery App", layout="wide", page_icon="🧘")

# ==========================================
# 1. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('wfh_burnout_dataset.csv')
    return df

df = load_data()

# MUST exactly match the order and names of UI inputs
cluster_features = [
    'work_hours', 'screen_time_hours', 'meetings_count', 'breaks_taken', 
    'after_hours_work', 'app_switches', 'sleep_hours', 'task_completion', 
    'isolation_index', 'fatigue_score'
]

# ==========================================
# 2. CLUSTERING UTILITIES
# ==========================================
@st.cache_resource
def train_clustering_model(data, features, n_clusters=3):
    """
    Fits and returns the Scaler and KMeans model without mutating global variables.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)
    
    return scaler, kmeans

@st.cache_data
def get_cluster_info(_scaler, _kmeans, data, features, n_clusters=3):
    """
    Extracts explicit data outputs (labels, centroids, profiles) safely.
    Uses _ underscores for parameters that shouldn't be hashed by Streamlit caching.
    """
    labels = _kmeans.labels_
    
    centroids = _scaler.inverse_transform(_kmeans.cluster_centers_)
    centroid_df = pd.DataFrame(centroids, columns=features)
    
    cluster_profiles = {}
    for i in range(n_clusters):
        work = centroid_df.loc[i, 'work_hours']
        sleep = centroid_df.loc[i, 'sleep_hours']
        fatigue = centroid_df.loc[i, 'fatigue_score']
        
        # Detailed naming and 6-month plans based on heuristics
        if fatigue > 6.5 and work > 9:
            name = "The Overworked & Exhausted"
            sugg = """🚨 **Status Review:** You are pushing dangerous hours and experiencing high fatigue. Immediate action is needed.
            
📅 **6-Month Improvement Plan:**
* **Months 1-2:** **Establish Boundaries.** Set a hard stop at 8 working hours. No after-hours emails. Start practicing a 30-minute wind-down routine before sleep without screens.
* **Months 3-4:** **Rebalance & Delegate.** Target consistent 7.5+ hour sleep schedules. Begin taking 10-minute micro-breaks every 90 minutes of work. If workload is impossible, actively delegate or drop non-essential meetings.
* **Months 5-6:** **Sustainable Rhythm.** Your fatigue should be dropping. Reintroduce a hobby or daily exercise. Maintain strict barriers between 'home' space and 'work' space to permanently lower burnout risk."""
        elif fatigue < 5 and sleep > 7:
            name = "The Balanced Worker"
            sugg = """✅ **Status Review:** Excellent! You maintain a strong functional balance between rest and work without sacrificing health.
            
📅 **6-Month Improvement Plan:**
* **Months 1-2:** **Solidify Routine.** Ensure your current sleep/work schedule is documented and protected from creeping meeting hours. 
* **Months 3-4:** **Optimize Focus.** Try batching your tasks to avoid app-switching fatigue, aiming for "deep work" blocks of 2 hours, followed by substantial offline breaks.
* **Months 5-6:** **Prevent Stagnation.** Having low burnout is great, but ensure you remain engaged. Pick up a low-stress learning opportunity or mentor someone else on how to achieve this balance!"""
        elif sleep < 6.5 and work <= 9:
            name = "The Sleep-Deprived"
            sugg = """🛌 **Status Review:** While your work hours are manageable, your sleep duration is unhealthy. This limits your recovery and fuels background fatigue.

📅 **6-Month Improvement Plan:**
* **Months 1-2:** **Sleep Hygiene.** Move all screens out of the bedroom. Go to bed 15 minutes earlier every week until you hit a 7-8 hour window. Reduce caffeine after 2 PM.
* **Months 3-4:** **Morning Optimization.** With better sleep, start a consistent morning routine before logging into work. Use the morning for 20 minutes of natural sunlight to reset your circadian rhythm.
* **Months 5-6:** **Performance Review.** As sleep stabilizes, you should notice sharper focus and faster task completion. Track if your App Switches decrease and actively defend your new sleep schedule."""
        elif work > 9 and fatigue <= 6.5:
            name = "The High-Achiever"
            sugg = """🔥 **Status Review:** You are working long hours but somehow managing the fatigue—for now. This phase is often the precursor to sudden burnout.

📅 **6-Month Improvement Plan:**
* **Months 1-2:** **Strategic Pullback.** Reduce meetings by 20% (decline non-essential invites). You are overworking; aim to shave 1 hour off your workday immediately by prioritizing high-impact tasks.
* **Months 3-4:** **Active Recovery.** Introduce forcing functions: schedule a mandatory 45-minute lunch away from the desk. Start logging off immediately when your core tasks are done, rather than 'finding' more work to fill the time.
* **Months 5-6:** **Efficiency over Volume.** Reduce work hours to 8.5/day. Focus entirely on task completion rates rather than time spent at the desk. You will find you produce the same quality in less time."""
        else:
            name = "The Moderate / Undefined"
            sugg = """☕ **Status Review:** You fall into a middle-ground pattern. Your habits aren't severely dangerous, but there is room to optimize your daily energy.

📅 **6-Month Improvement Plan:**
* **Months 1-2:** **Audit Your Days.** Spend 2 weeks tracking what exactly makes you tired (Is it specific meetings? App switching? Evening emails?). Start minimizing those specific triggers.
* **Months 3-4:** **Introduce Rhythm.** Adopt the Pomodoro technique (25 mins work, 5 mins rest) to ensure you aren't staring at screens for hours uninterrupted. Seek out 1 daily social interaction to combat isolation.
* **Months 5-6:** **Targeted Adjustment.** Depending on how your audit went, focus either on raising your sleep by 30 minutes or dropping your daily working hours by 45 minutes to find your personal sweet spot."""
            
        cluster_profiles[i] = {"name": f"Type {i}: {name}", "suggestion": sugg}
        
    return labels, centroid_df, cluster_profiles

def perform_clustering(data, features, n_clusters):
    """
    Wrapper function to provide all 5 variables cleanly as requested.
    """
    scaler, kmeans = train_clustering_model(data, features, n_clusters)
    labels, centroid_df, profiles = get_cluster_info(scaler, kmeans, data, features, n_clusters)
    return scaler, kmeans, labels, centroid_df, profiles


# ==========================================
# 3. UI TABS & HIGH-LEVEL STRUCTURE
# ==========================================
st.title("🧘 Work-From-Home Burnout & Suggestions App")
st.markdown("Analyze WFH habits, discover different *person types*, and get tailored recovery suggestions.")

tab1, tab2, tab3, tab4 = st.tabs(["🗂️ Data Explorer", "🔍 Explore Clusters", "👤 Client Matcher", "🗓️ Calendar Analyzer"])

with tab1:
    st.header("Dataset Overview")
    st.dataframe(df.head(10))
    
    st.subheader("Fatigue vs. Burnout Score")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=df, x='fatigue_score', y='burnout_score', hue='day_type', ax=ax)
    st.pyplot(fig)


with tab2:
    st.header("Discovering Person Types")
    st.markdown("Explore cluster solutions interactively.")
    
    n_clusters = st.slider("Number of Person Types to Discover (K)", min_value=2, max_value=6, value=3)
    
    # 1. Provide all variables cleanly via the unified wrapper
    scaler_temp, kmeans_temp, labels_temp, centroids_temp, profiles_temp = perform_clustering(df, cluster_features, n_clusters)
    
    # 2. Create explicit local copy to mutate and visualize
    temp_df = df.copy()
    temp_df['Cluster'] = labels_temp  # strictly separated from global df
    temp_df['Person Type'] = temp_df['Cluster'].apply(lambda x: profiles_temp[x]['name'])
    
    # 3. Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=temp_df, x='work_hours', y='fatigue_score', hue='Person Type', ax=ax, palette='Set2')
    st.pyplot(fig)
    
    st.subheader("Cluster Centroids (Averages)")
    st.dataframe(centroids_temp.style.highlight_max(axis=0, color='lightcoral').highlight_min(axis=0, color='lightgreen'))


with tab3:
    st.header("👤 Discover Your Person Type")
    st.markdown("Input your current habits to see which group you belong to and get tailored suggestions.")
    
    # Use fixed final model K=3 for predictions in this tab
    final_k = 3
    final_scaler, final_kmeans, _, _, final_profiles = perform_clustering(df, cluster_features, final_k)
    
    col1, col2 = st.columns(2)
    with col1:
        c_work = st.slider("Work Hours", 0.0, 16.0, 8.0)
        c_screen = st.slider("Screen Time (Hours)", 0.0, 16.0, 7.0)
        c_meet = st.slider("Number of Meetings", 0, 15, 3)
        c_breaks = st.slider("Breaks Taken", 0, 10, 3)
        c_tasks = st.slider("Task Completion (%)", 0.0, 100.0, 80.0)
    with col2:
        c_after = st.checkbox("Work after hours?", value=False)
        c_switches = st.slider("App Switches", 0, 150, 50)
        c_sleep = st.slider("Sleep Hours", 3.0, 12.0, 7.0)
        c_iso = st.slider("Isolation Index (1-10)", 1, 10, 5)
        c_fatigue = st.slider("Fatigue Score (0-10)", 0.0, 10.0, 5.0)
        
    if st.button("Analyze My Habits"):
        after_val = 1 if c_after else 0
        
        # Structure dataframe inputs EXACTLY in the feature list order
        raw_input_dict = {
            'work_hours': [c_work],
            'screen_time_hours': [c_screen],
            'meetings_count': [c_meet],
            'breaks_taken': [c_breaks],
            'after_hours_work': [after_val],
            'app_switches': [c_switches],
            'sleep_hours': [c_sleep],
            'task_completion': [c_tasks],
            'isolation_index': [c_iso],
            'fatigue_score': [c_fatigue]
        }
        
        input_data = pd.DataFrame(raw_input_dict)
        input_data = input_data[cluster_features] # Forces order matching to prevent scikit warnings
        
        # Scale and predict
        scaled_input = final_scaler.transform(input_data)
        cluster_assignment = final_kmeans.predict(scaled_input)[0]
        
        assigned_profile = final_profiles[cluster_assignment]
        
        st.success(f"### You are heavily matched with: **{assigned_profile['name']}**")
        st.markdown(assigned_profile['suggestion'])


with tab4:
    st.header("🗓️ Calendar Analyzer")
    st.markdown("Upload your `.ics` calendar file to analyze your schedule and get simple, smart, prioritized suggestions to improve your day.")
    
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        earliest_leave_time = st.time_input("Earliest realistic time you can leave work", value=time(17, 0))
    with col_b:
        commute_time = st.slider("How long does it take you to get home from work? (minutes)", 0, 120, 10)
    with col_c:
        show_more = st.checkbox("Show more recommendations", value=False)
    
    uploaded_file = st.file_uploader("Upload a Calendar File (.ics)", type=["ics"])
    if uploaded_file is not None:
        try:
            cal = Calendar.from_ical(uploaded_file.read())
            
            events = []
            for component in cal.walk():
                if component.name == "VEVENT":
                    summary = component.get('summary')
                    dtstart_val = component.get('dtstart')
                    dtend_val = component.get('dtend')
                    
                    if dtstart_val and dtend_val:
                        dtstart = dtstart_val.dt
                        dtend = dtend_val.dt
                        
                        # Only keep events with datetime
                        if isinstance(dtstart, datetime) and isinstance(dtend, datetime):
                            # Make naive for simplified standard math
                            dtstart = dtstart.replace(tzinfo=None)
                            dtend = dtend.replace(tzinfo=None)
                            duration = (dtend - dtstart).total_seconds() / 60.0
                            
                            events.append({
                                'Event': str(summary),
                                'Start': dtstart,
                                'End': dtend,
                                'Duration (mins)': duration,
                                'Date': dtstart.date()
                            })
            
            if len(events) == 0:
                st.warning("No timed events found in the calendar.")
            else:
                event_df = pd.DataFrame(events)
                event_df = event_df.sort_values('Start').reset_index(drop=True)
                
                st.subheader("📝 Extracted Events")
                display_df = event_df.copy()
                display_df['Start'] = display_df['Start'].dt.strftime('%H:%M')
                display_df['End'] = display_df['End'].dt.strftime('%H:%M')
                st.dataframe(display_df[['Date', 'Event', 'Start', 'End', 'Duration (mins)']])
                
                # Setup Display Limits
                max_recs = 6 if show_more else 3
                
                st.subheader("💡 Prioritized Analysis & Recommendations")
                
                dates = event_df['Date'].unique()
                for d in dates:
                    st.markdown(f"**Analysis for {d}:**")
                    day_events = event_df[event_df['Date'] == d].copy()
                    day_events = day_events.sort_values('Start')
                    
                    recs = []
                    
                    # --- RULE 1: Lunch Break Check ---
                    lunch_start = datetime.combine(d, datetime.min.time()).replace(hour=12)
                    lunch_end = datetime.combine(d, datetime.min.time()).replace(hour=14)
                    has_lunch_break = False
                    last_end_lunch = lunch_start
                    
                    for _, row in day_events.iterrows():
                        ev_start = row['Start']
                        if ev_start > last_end_lunch and ev_start < lunch_end:
                            if (ev_start - last_end_lunch).total_seconds() / 60.0 >= 30:
                                has_lunch_break = True
                        if row['End'] > last_end_lunch:
                            last_end_lunch = row['End']
                    if last_end_lunch < lunch_end:
                        if (lunch_end - last_end_lunch).total_seconds() / 60.0 >= 30:
                            has_lunch_break = True
                            
                    if not has_lunch_break:
                        recs.append({"priority": 95, "category": "Break", "message": "No proper lunch break scheduled → please guard at least 30 uninterrupted minutes between 12:00 and 14:00.", "type": "warning"})

                    # --- RULE 2 & 3: Meeting Overloads ---
                    total_meeting_time = day_events['Duration (mins)'].sum()
                    if total_meeting_time > 240: # 4 hours
                        recs.append({"priority": 90, "category": "Overload", "message": "Your day is meeting-heavy (>4 hrs) → avoid cognitively demanding tasks after meetings.", "type": "warning"})
                    
                    if len(day_events) >= 5:
                        recs.append({"priority": 85, "category": "Overload", "message": f"You have {len(day_events)} meetings today → consider deferring non-essential ones.", "type": "warning"})
                        
                    # --- RULE 4: Long Meetings ---
                    if (day_events['Duration (mins)'] > 90).any():
                        recs.append({"priority": 80, "category": "Break", "message": "You have a long meeting (>90 mins) → take a break afterward.", "type": "warning"})

                    # --- RULE 5: Late Meetings ---
                    late_meetings = day_events[day_events['End'].dt.hour >= 17]
                    if len(late_meetings) > 0:
                        recs.append({"priority": 75, "category": "Balance", "message": "You have meetings scheduled after 17:00 → ensure you maintain a strict hard stop outwards.", "type": "warning"})

                    # --- INTER-MEETING RULES (B2B, Gaps) ---
                    has_b2b = False
                    for i in range(len(day_events) - 1):
                        curr_end = day_events.iloc[i]['End']
                        next_start = day_events.iloc[i+1]['Start']
                        gap = (next_start - curr_end).total_seconds() / 60.0
                        
                        if 0 <= gap < 10:
                            has_b2b = True
                            
                        # Free Time / Focus / Hybrid
                        if gap >= 60:
                            recs.append({"priority": 65, "category": "Focus", "message": f"You have a large free block between {curr_end.strftime('%H:%M')} and {next_start.strftime('%H:%M')} → good for focused work.", "type": "success"})
                            if gap >= 90 and curr_end.hour >= 14:
                                recs.append({"priority": 55, "category": "Hybrid", "message": "You have large free time late in the day → you could leave earlier, rest, and finish remaining work later at home.", "type": "info"})
                                
                        # Email Windows
                        elif 30 <= gap < 60:
                            recs.append({"priority": 60, "category": "Email", "message": f"You have a short gap between {curr_end.strftime('%H:%M')} and {next_start.strftime('%H:%M')} → perfect window for processing emails.", "type": "info"})
                            
                    if has_b2b:
                        recs.append({"priority": 80, "category": "Break", "message": "You have consecutive meetings → consider adding short breaks.", "type": "warning"})
                        
                    # --- RULE 6: Start of Day / End of Day / Commute Logic ---
                    if len(day_events) > 0:
                        working_start = datetime.combine(d, datetime.min.time()).replace(hour=9)
                        first_meeting_start = day_events.iloc[0]['Start']
                        
                        if first_meeting_start > working_start:
                            gap = (first_meeting_start - working_start).total_seconds() / 60.0
                            if gap >= 60:
                                recs.append({"priority": 65, "category": "Focus", "message": f"Your morning is entirely clear until {first_meeting_start.strftime('%H:%M')} → great time for uninterrupted deep work.", "type": "success"})

                        last_meeting = day_events.iloc[-1]
                        last_end = last_meeting['End']
                        
                        # No afternoon meetings check (afternoon defined as 13:00 / 1 PM onwards)
                        if last_end.hour < 13:
                            recs.append({"priority": 70, "category": "Focus", "message": "You do not have meetings in the afternoon, so this is a good time for focused work or completing important tasks.", "type": "success"})
                        
                        # Earliest leave time logic
                        leave_datetime = datetime.combine(d, earliest_leave_time)
                        
                        if last_end >= leave_datetime:
                            arrival_time = last_end + pd.Timedelta(minutes=commute_time)
                            recs.append({"priority": 50, "category": "Commute", "message": f"Your last meeting ends at {last_end.strftime('%H:%M')}. With a commute of {commute_time} minutes, you would arrive home around {arrival_time.strftime('%H:%M')}.", "type": "warning"})
                        else:
                            gap_to_leave = (leave_datetime - last_end).total_seconds() / 60.0
                            if gap_to_leave <= 45:
                                recs.append({"priority": 60, "category": "Commute", "message": f"You have no meetings after {last_end.strftime('%H:%M')}. If your tasks are complete and your work is flexible, leaving around {earliest_leave_time.strftime('%H:%M')} could be realistic.", "type": "info"})
                            else:
                                recs.append({"priority": 58, "category": "Focus", "message": f"You have no meetings after {last_end.strftime('%H:%M')}. Use the remaining time until {earliest_leave_time.strftime('%H:%M')} for focused work, admin tasks, or planning.", "type": "info"})
                        
                    # --- FILTER & RANK ---
                    # Sort desc by priority
                    recs = sorted(recs, key=lambda x: x['priority'], reverse=True)
                    
                    final_recs = []
                    seen_messages = set()
                    
                    for r in recs:
                        if r['message'] in seen_messages:
                            continue
                            
                        # Limit to max 2 recommendations of the same category to ensure diversity
                        cat_count = sum([1 for fr in final_recs if fr['category'] == r['category']])
                        if cat_count < 2:
                            final_recs.append(r)
                            seen_messages.add(r['message'])
                            
                    # Apply length restraint
                    final_recs_trimmed = final_recs[:max_recs]
                    
                    # Positive Affirmation if healthy schedule
                    has_warnings = any(r['type'] == 'warning' for r in final_recs_trimmed)
                    if not has_warnings:
                        st.success("✨ Your schedule looks very balanced and manageable today! No major overloads detected.")
                    
                    # Render Recommendations
                    for r in final_recs_trimmed:
                        if r['type'] == 'warning':
                            st.warning(f"**{r['category']}:** {r['message']}")
                        elif r['type'] == 'success':
                            st.success(f"**{r['category']}:** {r['message']}")
                        else:
                            st.info(f"**{r['category']}:** {r['message']}")
                            
                    if not final_recs_trimmed and has_warnings == False:
                        st.info("Enjoy your calm day!")

                    st.markdown("---")

        except Exception as e:
            st.error(f"Error parsing the calendar file: {e}")
