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

st.markdown("""
<style>
    /* Global App Settings & Spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    
    hr {
        margin: 1.5rem 0;
        border-color: #e5e7eb;
    }
    
    /* Hero/Header Section */
    .hero-container {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .hero-container h1 {
        font-size: 2.3rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
        color: #ffffff;
    }
    .hero-container p {
        font-size: 1.15rem;
        color: #d1d5db;
        margin-bottom: 0;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding-bottom: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f3f4f6;
        border-radius: 6px 6px 0 0;
        padding: 10px 24px;
        color: #4b5563;
        font-weight: 600;
        border: 1px solid transparent;
        transition: background-color 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e5e7eb;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #2563eb !important;
        border-bottom: 3px solid #2563eb;
        border-top: 1px solid #e5e7eb;
        border-left: 1px solid #e5e7eb;
        border-right: 1px solid #e5e7eb;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.02);
    }

    /* Dashboard Cards (Metrics) */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #f3f4f6;
    }
    div[data-testid="stMetricValue"] {
        color: #111827;
        font-weight: 700;
        font-size: 1.8rem;
    }
    div[data-testid="stMetricLabel"] {
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
    }

    /* Recommendation Alerts UI */
    .stAlert {
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    
    /* Section Typography */
    h2, h3 {
        color: #1f2937 !important;
        font-weight: 700 !important;
        margin-top: 1rem !important;
    }
    /* Visual Schedule Agenda */
    .agenda-container {
        margin: 1rem 0 1.5rem 0;
    }
    .agenda-card {
        border-left: 5px solid #3b82f6;
        background-color: #f8fafc;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .agenda-card.short { border-left-color: #10b981; }
    .agenda-card.long { border-left-color: #ef4444; }
    
    .agenda-time {
        font-weight: 600;
        color: #4b5563;
        font-size: 0.9rem;
    }
    .agenda-title {
        font-weight: 700;
        color: #111827;
        font-size: 1.05rem;
        margin: 4px 0;
    }
    .agenda-duration {
        font-size: 0.8rem;
        color: #6b7280;
    }
    .agenda-free {
        text-align: center;
        color: #6b7280;
        font-style: italic;
        padding: 8px 0;
        font-size: 0.9rem;
        border-right: 2px dashed #d1d5db;
        border-left: 2px dashed #d1d5db;
        margin: 8px auto;
        width: 80%;
        background-color: #fcfcfc;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

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
    'work_hours', 'meetings_count', 'breaks_taken', 
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
st.markdown("""
<div class="hero-container">
    <h1> Work-From-Home Burnout & Suggestions App</h1>
    <p>Analyze WFH habits, discover different <strong>person types</strong>, and get tailored recovery suggestions.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗂️ Data Explorer", "🔍 Explore Clusters", "👤 Client Matcher", "🗓️ Calendar Analyzer", "⌚ Wearable Preview"])

with tab1:
    with st.container():
        st.header("📊 Behavior Insights Dashboard")
        st.markdown("Explore key trends and behavioral patterns in the remote work dataset.")
        
        # -----------------------------------
        # 1. Dataset Overview
        # -----------------------------------
        st.subheader("📌 Dataset Overview")
        num_rows = df.shape[0]
        num_cols = df.shape[1]
        num_numeric = len(df.select_dtypes(include=[np.number]).columns)
        num_cat = len(df.select_dtypes(exclude=[np.number]).columns)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", num_rows)
        c2.metric("Total Columns", num_cols)
        c3.metric("Numerical Vars", num_numeric)
        c4.metric("Categorical Vars", num_cat)
        
        with st.expander("Show raw dataset preview"):
            st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")

        # -----------------------------------
        # 2. Data Quality Snapshot
        # -----------------------------------
        st.subheader("🛡️ Data Quality Snapshot")
        missing_count = df.isnull().sum().sum()
        duplicate_count = df.duplicated().sum()
        
        dq1, dq2 = st.columns(2)
        if missing_count > 0:
            dq1.warning(f"⚠️ **{missing_count}** missing values detected.")
        else:
            dq1.success("✅ **0** missing values detected.")
            
        if duplicate_count > 0:
            dq2.warning(f"⚠️ **{duplicate_count}** duplicate rows detected.")
        else:
            dq2.success("✅ **0** duplicate rows detected.")

        st.markdown("---")

        # -----------------------------------
        # 3. Key Behavioral Patterns
        # -----------------------------------
        st.subheader("📈 Key Behavioral Patterns")
        
        p1, p2 = st.columns(2)
        
        with p1:
            st.markdown("**A. Work Hours Distribution**")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.histplot(data=df, x='work_hours', bins=20, kde=True, color='#3b82f6', ax=ax1)
            ax1.set_xlabel("Work Hours")
            ax1.set_ylabel("Count")
            st.pyplot(fig1)
            
            st.markdown("**C. Fatigue vs Burnout**")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df, x='fatigue_score', y='burnout_score', alpha=0.6, color='#ef4444', ax=ax3)
            ax3.set_xlabel("Fatigue Score")
            ax3.set_ylabel("Burnout Score")
            st.pyplot(fig3)

        with p2:
            st.markdown("**B. Sleep Hours Distribution**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.histplot(data=df, x='sleep_hours', bins=20, kde=True, color='#10b981', ax=ax2)
            ax2.set_xlabel("Sleep Hours")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

            st.markdown("**D. Work Hours vs Fatigue**")
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df, x='work_hours', y='fatigue_score', alpha=0.6, color='#f59e0b', ax=ax4)
            ax4.set_xlabel("Work Hours")
            ax4.set_ylabel("Fatigue Score")
            st.pyplot(fig4)

        st.markdown("---")

        # -----------------------------------
        # 4. Key Insights Section
        # -----------------------------------
        st.subheader("💡 Key Insights Summary")
        
        avg_work = df['work_hours'].mean()
        avg_sleep = df['sleep_hours'].mean()
        avg_fatigue = df['fatigue_score'].mean()
        avg_burnout = df['burnout_score'].mean()
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Avg Work Hours", f"{avg_work:.1f} hr")
        k2.metric("Avg Sleep Hours", f"{avg_sleep:.1f} hr")
        k3.metric("Avg Fatigue", f"{avg_fatigue:.1f} / 10")
        k4.metric("Avg Burnout", f"{avg_burnout:.1f} / 10")
        
        st.info("📌 **Observation:** Higher work hours are intrinsically linked to elevated fatigue. Guaranteeing 7+ hours of sleep per night acts as the most aggressive buffer against compounding burnout scores.")

with tab2:
    with st.container():
        st.header("👥 Person Profiles Overview")
        st.markdown("Based on the data, employees fall into **3 distinct behavioral types**. Below is the final model categorization.")
        
        # Always use fixed final model K=3
        final_k = 3
        scaler_temp, kmeans_temp, labels_temp, centroids_temp, profiles_temp = perform_clustering(df, cluster_features, final_k)
        
        # -----------------------------------
        # 1. Section: Person Profiles Overview
        # -----------------------------------
        for idx in range(final_k):
            profile = profiles_temp[idx]
            with st.expander(f"**{profile['name']}**", expanded=True):
                st.markdown(profile['suggestion'])

        st.markdown("---")
        
        # -----------------------------------
        # 2. Section: Cluster Characteristics
        # -----------------------------------
        st.subheader("📊 Cluster Characteristics")
        st.markdown("Average behaviors per profile type:")
        
        display_features = ['work_hours', 'sleep_hours', 'fatigue_score', 'burnout_score']
        display_features = list(dict.fromkeys(display_features))
        
        stat_df = df.copy()
        stat_df['Cluster'] = labels_temp
        
        valid_display_features = [col for col in display_features if col in stat_df.columns]
        
        cluster_summary = stat_df.groupby('Cluster')[valid_display_features].mean()
        cluster_summary.index = [profiles_temp[i]['name'].split(':')[1].strip() for i in range(final_k)]
        
        if cluster_summary.columns.is_unique and cluster_summary.index.is_unique:
            st.dataframe(cluster_summary.style.highlight_max(axis=0, color='#fca5a5').highlight_min(axis=0, color='#bbf7d0'), use_container_width=True)
        else:
            st.dataframe(cluster_summary, use_container_width=True)

        st.markdown("---")
        
        # -----------------------------------
        # 3. Section: Cluster Visualization & Comparison
        # -----------------------------------
        st.subheader("🔍 Cluster Visualization")
        
        temp_df = df.copy()
        temp_df['Cluster'] = labels_temp
        temp_df['Person Type'] = temp_df['Cluster'].apply(lambda x: profiles_temp[x]['name'].split(':')[1].strip())
        
        v_col1, v_col2 = st.columns([1, 2])
        with v_col1:
            x_axis = st.selectbox("X-Axis", display_features, index=0)
            y_axis = st.selectbox("Y-Axis", display_features, index=2)
            
            st.markdown("### ⚖️ Profile Comparison")
            st.info("🔹 **Type 0:** Tend to show elevated fatigue linked with highest average work hours.")
            st.success("🔹 **Type 1:** Balanced habits resulting in strong sleep-to-fatigue ratios.")
            st.warning("🔹 **Type 2:** Sleep deprived patterns dragging up background exhaustion.")

        with v_col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.scatterplot(data=temp_df, x=x_axis, y=y_axis, hue='Person Type', ax=ax, palette=['#3b82f6', '#10b981', '#f59e0b'])
            ax.set_xlabel(x_axis.replace('_', ' ').title())
            ax.set_ylabel(y_axis.replace('_', ' ').title())
            st.pyplot(fig)


with tab3:
    st.header("👤 Discover Your Person Type")
    st.markdown("Input your current habits to see which group you belong to and get tailored suggestions.")
    
    # Use fixed final model K=3 for predictions in this tab
    final_k = 3
    final_scaler, final_kmeans, _, _, final_profiles = perform_clustering(df, cluster_features, final_k)
    
    col1, col2 = st.columns(2)
    with col1:
        c_work = st.slider("Work Hours", 0.0, 16.0, 8.0)
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
        distances = final_kmeans.transform(scaled_input)[0]
        
        # Inverse distance probability for a simple confidence %
        inv_dists = 1 / (distances + 1e-5)
        confidence = (inv_dists[cluster_assignment] / np.sum(inv_dists)) * 100
        
        assigned_profile = final_profiles[cluster_assignment]
        profile_name = assigned_profile['name'].split(':')[1].strip()
        
        # Get centroid data directly from the previously stored clustering wrappers
        _, _, _, final_centroids, _ = perform_clustering(df, cluster_features, final_k)
        profile_avgs = final_centroids.iloc[cluster_assignment]
        
        st.markdown("---")
        st.subheader("🎯 Your Work Profile Analysis")
        
        # UI aesthetic matching the Person Profiles
        st.markdown(f'''
        <div style="background-color: #f8fafc; border-left: 5px solid #3b82f6; border-radius: 8px; padding: 1.5rem; margin-top: 1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <h2 style="margin-top: 0; color: #111827;">{profile_name}</h2>
            <p style="font-size: 1.1rem; color: #4b5563; margin-bottom: 0;"><strong>Match Confidence:</strong> <span style="color: #10b981; font-weight: bold;">{confidence:.1f}%</span></p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("#### 🔍 Why You Matched")
            st.markdown("Your key behavioral inputs mapped aggressively to this profile's centroid. Here is a breakdown of your key contributors vs the profile average:")
            
            critical_vars = ['work_hours', 'sleep_hours', 'fatigue_score', 'meetings_count']
            
            for var in critical_vars:
                usr_val = input_data[var].iloc[0]
                avg_val = profile_avgs[var]
                
                diff = usr_val - avg_val
                trend = "↑" if diff > 0 else "↓" if diff < 0 else "="
                # Simple logic for bad/good colors based purely on typical productivity logic
                c_color = "red" if (var in ['work_hours', 'fatigue_score'] and diff > 0) or (var == 'sleep_hours' and diff < 0) else "green"
                
                st.markdown(f"**{var.replace('_', ' ').title()}:**")
                st.markdown(f"↳ You: `{usr_val:.1f}` | Profile Avg: `{avg_val:.1f}` <span style='color:{c_color}; font-weight:bold;'>{trend}</span>", unsafe_allow_html=True)
                
        with col_m2:
            st.markdown("#### 💡 Tailored Action Plan")
            st.info(f"As a recognized **{profile_name}**, the model specifically prioritizes the following intervention plan for your psychological safety:")
            st.markdown(assigned_profile['suggestion'])


with tab4:
    with st.container():
        st.header("🗓️ Calendar Analyzer")
        st.markdown("Upload your `.ics` calendar file for a structured, professional overview of your week and actionable daily insights.")
    
    # -----------------------------------
    # 4. Preferred Working Hours
    # -----------------------------------
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        preferred_start_time = st.time_input("Preferred Start Time", value=time(9, 0))
    with col_b:
        earliest_leave_time = st.time_input("Earliest realistic time you can leave work", value=time(17, 0))
    with col_c:
        commute_time = st.slider("How long does it take you to get home? (minutes)", 0, 120, 10)
    
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
                dates = event_df['Date'].unique()
                num_days = len(dates)
                
                # Global week aggregations
                total_meetings_week = len(event_df)
                total_hours_week = event_df['Duration (mins)'].sum() / 60.0
                late_meetings_count = len(event_df[event_df['End'].dt.time > earliest_leave_time])
                
                meetings_per_day = event_df.groupby('Date').size()
                busiest_day_date = meetings_per_day.idxmax()
                best_day_date = meetings_per_day.idxmin()
                
                total_free_mins = 0
                large_blocks_count = 0
                short_gaps_count = 0
                
                # Pre-calculate global free time metrics
                for d in dates:
                    day_events = event_df[event_df['Date'] == d].sort_values('Start')
                    day_start = datetime.combine(d, preferred_start_time)
                    day_end = datetime.combine(d, earliest_leave_time)
                    
                    last_end_tracker = day_start
                    for _, row in day_events.iterrows():
                        ev_start = max(day_start, row['Start'])
                        ev_end = min(day_end, row['End'])
                        if ev_start > ev_end: 
                            ev_start, ev_end = ev_end, ev_start
                            
                        if ev_start > last_end_tracker:
                            gap = (ev_start - last_end_tracker).total_seconds() / 60.0
                            total_free_mins += gap
                            if gap >= 60: large_blocks_count += 1
                            if 0 < gap < 30: short_gaps_count += 1
                            
                        last_end_tracker = max(last_end_tracker, ev_end)
                        
                    if last_end_tracker < day_end:
                        gap = (day_end - last_end_tracker).total_seconds() / 60.0
                        total_free_mins += gap
                        if gap >= 60: large_blocks_count += 1
                        if 0 < gap < 30: short_gaps_count += 1

                # Calculate Scores (0-10) heuristically
                focus_score = round(min(10.0, (large_blocks_count / max(1, num_days)) * 3), 1)
                fragmentation_score = round(min(10.0, (short_gaps_count / max(1, num_days)) * 2), 1)

                # -----------------------------------
                # 1. Weekly Overview
                # -----------------------------------
                with st.container():
                    st.subheader("📊 Weekly Overview")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Meetings", total_meetings_week)
                    col2.metric("Total Meeting Time", f"{total_hours_week:.1f}h")
                    col3.metric("Total Free Time", f"{int(total_free_mins//60)}h {int(total_free_mins%60)}m")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    col4, col5, col6 = st.columns(3)
                    col4.metric("Focus Time Score", f"{focus_score} / 10")
                    col5.metric("Fragmentation Score", f"{fragmentation_score} / 10")
                    col6.metric("Late Meetings", late_meetings_count)
                    
                    st.markdown("---")

                # -----------------------------------
                # 2. Weekly Insights
                # -----------------------------------
                with st.container():
                    st.subheader("💡 Weekly Insights")
                    weekly_recs = []
                
                if total_hours_week > 15:
                    weekly_recs.append({"priority": 90, "category": "Overload", "message": f"You have {total_hours_week:.1f} hours of meetings this week. Prioritize delegating or declining non-essential invites.", "type": "warning"})
                
                if late_meetings_count >= 3:
                    weekly_recs.append({"priority": 85, "category": "Balance", "message": f"You have {late_meetings_count} meetings extending past your earliest leave time ({earliest_leave_time.strftime('%H:%M')}). Try to enforce a harder evening boundary.", "type": "warning"})
                    
                weekly_recs.append({"priority": 80, "category": "Overload", "message": f"Your busiest day is {busiest_day_date.strftime('%A')} ({meetings_per_day.max()} meetings). Prepare in advance or avoid scheduling heavy tasks immediately before/after.", "type": "warning"})
                weekly_recs.append({"priority": 75, "category": "Focus", "message": f"{best_day_date.strftime('%A')} is your lightest day ({meetings_per_day.min()} meetings) → guard this day aggressively for deep focus work.", "type": "success"})
                
                weekly_recs = sorted(weekly_recs, key=lambda x: x['priority'], reverse=True)[:3]
                
                for r in weekly_recs:
                    if r['type'] == 'warning':
                        st.warning(f"**{r['category']}:** {r['message']}")
                    elif r['type'] == 'success':
                        st.success(f"**{r['category']}:** {r['message']}")
                    else:
                        st.info(f"**{r['category']}:** {r['message']}")
                
                if not any(r['type'] == 'warning' for r in weekly_recs):
                    st.success("✨ Your week looks structurally sound! Low overload risk detected.")

                st.markdown("---")

                # -----------------------------------
                # 3. Daily Breakdown
                # -----------------------------------
                with st.container():
                    st.subheader("📅 Daily Breakdown")
                
                for d in dates:
                    day_events = event_df[event_df['Date'] == d].copy()
                    day_events = day_events.sort_values('Start')
                    num_meetings = len(day_events)
                    day_name_str = d.strftime('%A, %d %B')
                    
                    with st.expander(f"{day_name_str} — {num_meetings} meeting{'s' if num_meetings != 1 else ''}"):
                        day_total_mins = day_events['Duration (mins)'].sum()
                        first_meet = day_events.iloc[0]['Start'].strftime('%H:%M') if num_meetings > 0 else "N/A"
                        last_meet = day_events.iloc[-1]['End'].strftime('%H:%M') if num_meetings > 0 else "N/A"
                        longest_meet_mins = day_events['Duration (mins)'].max() if num_meetings > 0 else 0
                        
                        # Find Free Time for the specific day between preferred start and earliest leave
                        day_start = datetime.combine(d, preferred_start_time)
                        day_end = datetime.combine(d, earliest_leave_time)
                        local_free_mins = 0
                        
                        last_e = day_start
                        for _, row in day_events.iterrows():
                            ev_s = max(day_start, row['Start'])
                            ev_e = min(day_end, row['End'])
                            if ev_s > ev_e: ev_s, ev_e = ev_e, ev_s
                            if ev_s > last_e:
                                gap = (ev_s - last_e).total_seconds() / 60.0
                                local_free_mins += gap
                            last_e = max(last_e, ev_e)
                        
                        if last_e < day_end:
                            gap = (day_end - last_e).total_seconds() / 60.0
                            local_free_mins += gap
                        
                        # Display Daily Metrics
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("Tot. Meet Time", f"{int(day_total_mins//60)}h {int(day_total_mins%60)}m")
                        c2.metric("Free Time", f"{int(local_free_mins//60)}h {int(local_free_mins%60)}m")
                        c3.metric("Longest Meet", f"{int(longest_meet_mins)}m")
                        c4.metric("First Start", first_meet)
                        c5.metric("Last End", last_meet)
                        
                        # Visual Agenda Display
                        st.markdown("##### 📝 Schedule")
                        
                        agenda_html = ['<div class="agenda-container">']
                        last_end_agenda = datetime.combine(d, preferred_start_time)
                        
                        for _, row in day_events.iterrows():
                            ev_start = row['Start']
                            ev_end = row['End']
                            title = row['Event']
                            dur = row['Duration (mins)']
                            
                            # Free block logic
                            if ev_start > last_end_agenda:
                                gap = (ev_start - last_end_agenda).total_seconds() / 60.0
                                if gap >= 30:
                                    agenda_html.append(
                                        '<div class="agenda-free">'
                                        f"☕ {last_end_agenda.strftime('%H:%M')} – {ev_start.strftime('%H:%M')} | {int(gap)} min Free Block"
                                        '</div>'
                                    )
                            
                            # Meeting card logic
                            color_class = "normal"
                            if dur < 30: color_class = "short"
                            elif dur > 90: color_class = "long"
                            
                            agenda_html.append(
                                f'<div class="agenda-card {color_class}">'
                                f'<div class="agenda-time">{ev_start.strftime("%H:%M")} – {ev_end.strftime("%H:%M")}</div>'
                                f'<div class="agenda-title">🔹 {title}</div>'
                                f'<div class="agenda-duration">🕒 {int(dur)} mins</div>'
                                '</div>'
                            )
                            
                            last_end_agenda = max(last_end_agenda, ev_end)
                        
                        # Free block logic to end of day boundary
                        day_end_agenda = datetime.combine(d, earliest_leave_time)
                        if last_end_agenda < day_end_agenda:
                            gap = (day_end_agenda - last_end_agenda).total_seconds() / 60.0
                            if gap >= 30:
                                agenda_html.append(
                                    '<div class="agenda-free">'
                                    f"🚀 {last_end_agenda.strftime('%H:%M')} – {day_end_agenda.strftime('%H:%M')} | {int(gap)} min Focus / Wrap-up"
                                    '</div>'
                                )

                        agenda_html.append('</div>')
                        st.markdown("".join(agenda_html), unsafe_allow_html=True)
                        
                        with st.expander("Show raw events table"):
                            display_df = day_events.copy()
                            display_df['Start'] = display_df['Start'].dt.strftime('%H:%M')
                            display_df['End'] = display_df['End'].dt.strftime('%H:%M')
                            st.dataframe(display_df[['Event', 'Start', 'End', 'Duration (mins)']], use_container_width=True)
                        
                        # Daily Recommendations
                        st.markdown("##### 💡 Daily Advice")
                        show_more_day = st.checkbox("Show more recommendations", key=f"show_more_{d}")
                        max_recs = 6 if show_more_day else 3
                        
                        recs = []
                        
                        # Rule 1: Lunch verification
                        lunch_start = datetime.combine(d, time(12, 0))
                        lunch_end = datetime.combine(d, time(14, 0))
                        has_lunch = False
                        l_track = lunch_start
                        for _, row in day_events.iterrows():
                            ev_start = row['Start']
                            if ev_start > l_track and ev_start <= lunch_end:
                                if (ev_start - l_track).total_seconds() / 60.0 >= 30:
                                    has_lunch = True
                            l_track = max(l_track, row['End'])
                        if l_track < lunch_end and (lunch_end - l_track).total_seconds() / 60.0 >= 30:
                            has_lunch = True
                            
                        if not has_lunch and num_meetings > 0:
                            recs.append({"priority": 95, "category": "Break", "message": "No proper lunch break scheduled → guard 30+ minutes between 12:00 and 14:00.", "type": "warning"})

                        # Rule 2: Overloads
                        if day_total_mins > 240:
                            recs.append({"priority": 90, "category": "Overload", "message": "Heavy meeting day (>4 hrs) → avoid demanding tasks immediately after meetings.", "type": "warning"})
                        if num_meetings >= 5:
                            recs.append({"priority": 85, "category": "Overload", "message": f"{num_meetings} meetings today → evaluate deferring non-essential ones.", "type": "warning"})
                            
                        # Rule 3: Buffer integration (long / back-to-back)
                        if longest_meet_mins > 90:
                            recs.append({"priority": 80, "category": "Buffer", "message": "You have a long meeting (>90 mins) → ensure you take a strict recovery buffer immediately afterward.", "type": "warning"})

                        has_b2b = False
                        for i in range(len(day_events) - 1):
                            curr_end = day_events.iloc[i]['End']
                            next_start = day_events.iloc[i+1]['Start']
                            gap = (next_start - curr_end).total_seconds() / 60.0
                            
                            # Back-to-back buffer
                            if 0 <= gap < 10:
                                has_b2b = True
                                
                            # Focus block / Email window (integrate directly)
                            if gap >= 60:
                                recs.append({"priority": 65, "category": "Focus", "message": f"Large free block from {curr_end.strftime('%H:%M')} to {next_start.strftime('%H:%M')} → reserve this exclusively for deep work.", "type": "success"})
                            elif 30 <= gap < 60:
                                recs.append({"priority": 60, "category": "Email", "message": f"Short gap from {curr_end.strftime('%H:%M')} to {next_start.strftime('%H:%M')} → optimal window for processing emails/admin.", "type": "info"})
                                
                        if has_b2b:
                            recs.append({"priority": 80, "category": "Buffer", "message": "You have consecutive back-to-back meetings → consider stepping away for short 5-minute transition breaks.", "type": "warning"})
                            
                        # Rule 4: Preferred bounds and commute
                        if num_meetings > 0:
                            first_m = day_events.iloc[0]['Start']
                            if first_m > day_start and (first_m - day_start).total_seconds() / 60.0 >= 60:
                                recs.append({"priority": 68, "category": "Focus", "message": f"Your morning is entirely clear until {first_m.strftime('%H:%M')} → excellent time for uninterrupted focus.", "type": "success"})

                            last_m = day_events.iloc[-1]['End']
                            
                            # No afternoon meetings check (After 13:00)
                            if last_m.hour < 13:
                                recs.append({"priority": 70, "category": "Focus", "message": "You do not have meetings in the afternoon, so this is a great time to complete important tasks or admin work.", "type": "success"})
                            
                            if last_m >= day_end:
                                arr_time = last_m + pd.Timedelta(minutes=commute_time)
                                recs.append({"priority": 50, "category": "Commute", "message": f"Last meeting runs until {last_m.strftime('%H:%M')}. With commute, you'd arrive home around {arr_time.strftime('%H:%M')}.", "type": "warning"})
                            else:
                                rem_mins = (day_end - last_m).total_seconds() / 60.0
                                if rem_mins <= 45:
                                    recs.append({"priority": 60, "category": "Commute", "message": f"No meetings after {last_m.strftime('%H:%M')}. Assuming tasks are complete, leaving around your earliest departure ({earliest_leave_time.strftime('%H:%M')}) is realistic.", "type": "info"})
                                else:
                                    recs.append({"priority": 58, "category": "Admin", "message": f"No meetings after {last_m.strftime('%H:%M')}. Use the remaining time until {earliest_leave_time.strftime('%H:%M')} for focused work or end-of-day planning.", "type": "info"})
                            
                        # Sort and trim daily recommendations
                        recs = sorted(recs, key=lambda x: x['priority'], reverse=True)
                        final_recs = []
                        seen_cats = set()
                        for r in recs:
                            # Filter to max 1 per category for diversity
                            if r['category'] not in seen_cats:
                                final_recs.append(r)
                                seen_cats.add(r['category'])
                        
                        final_recs_trimmed = final_recs[:max_recs]
                        
                        if not final_recs_trimmed:
                            st.success("✨ Your day is perfectly balanced. Enjoy!")
                        else:
                            for r in final_recs_trimmed:
                                if r['type'] == 'warning':
                                    st.warning(f"**{r['category']}:** {r['message']}")
                                elif r['type'] == 'success':
                                    st.success(f"**{r['category']}:** {r['message']}")
                                else:
                                    st.info(f"**{r['category']}:** {r['message']}")

                st.markdown("---")
                
                # -----------------------------------
                # 4. Behavioral Interpretation
                # -----------------------------------
                with st.container():
                    st.subheader("🧩 Behavioral Interpretation")
                    st.markdown("**Based on your weekly schedule patterns, your work rhythm is most similar to:**")
                    
                    # Rule-based heuristic match leveraging contextual week variables
                    if total_hours_week >= 15 and (focus_score < 4.0 or late_meetings_count >= 3):
                        matched_profile = "The Overworked & Exhausted"
                        match_reason = f"Your week shows a heavy meeting load ({total_hours_week:.1f} hours) with frequent interruptions, significantly reducing your opportunity for deep work."
                        sugg_1 = f"You have {late_meetings_count} late meetings this week. Consider enforcing a strict log-off boundary to avoid compounding exhaustion." if late_meetings_count > 0 else "Avoid scheduling complex tasks at the end of the day when background fatigue is highest."
                        sugg_2 = f"{best_day_date.strftime('%A')} is your lightest day. Guard this block aggressively for uninterrupted deep work to catch up without distractions."
                        match_border = "#ef4444"
                        match_bg = "#fef2f2"
                        match_icon = "🚨"
                        
                    elif focus_score >= 6.0 and total_hours_week <= 12 and late_meetings_count <= 1:
                        matched_profile = "The Balanced Worker"
                        match_reason = f"Your schedule maintains an excellent separation between collaboration and focus time."
                        sugg_1 = f"Maintain the boundaries you've set, especially on {best_day_date.strftime('%A')} where you have the highest focus efficiency."
                        sugg_2 = "Use short 10-minute buffers after intense meetings to avoid accumulating cognitive fatigue throughout the day."
                        match_border = "#10b981"
                        match_bg = "#f0fdf4"
                        match_icon = "✅"
                        
                    elif fragmentation_score >= 6.0 or (total_hours_week < 15 and late_meetings_count >= 2):
                        matched_profile = "The Sleep-Deprived / Fragmented"
                        match_reason = f"Your schedule is highly scattered. Even with fewer total hours, frequent context-switching prevents meaningful progress and raises overall exhaustion."
                        sugg_1 = f"Batch your meetings! For example, {busiest_day_date.strftime('%A')} is loaded with disjointed blocks. Consolidate those calls to open up contiguous free blocks."
                        sugg_2 = "Turn off notifications entirely during gaps shorter than 30 minutes to give your brain a true recovery transition."
                        match_border = "#f59e0b"
                        match_bg = "#fffbeb"
                        match_icon = "⚠️"
                        
                    else:
                        matched_profile = "The Flexible Worker"
                        match_reason = "Your week shows a balanced workload, but frequent interruptions or ad-hoc meetings may reduce your focus efficiency."
                        sugg_1 = f"Identify your best focus window on {best_day_date.strftime('%A')} (your lightest day) and permanently block it out directly in your calendar."
                        sugg_2 = "Evaluate if any of your recurring scattered meetings across the week can be consolidated to an afternoon."
                        match_border = "#3b82f6"
                        match_bg = "#eff6ff"
                        match_icon = "☕"
                    
                    st.markdown(f'''
                    <div style="background-color: {match_bg}; border-left: 5px solid {match_border}; border-radius: 8px; padding: 1.5rem; margin-top: 1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); color: #1f2937;">
                        <h3 style="margin-top: 0; color: #111827;">{match_icon} {matched_profile}</h3>
                        <p style="font-size: 1.05rem; margin-bottom: 1.5rem; line-height: 1.6;"><strong>Why?</strong> {match_reason}</p>
                        <h5 style="color: #374151; margin-bottom: 0.5rem; font-weight: 600;">Contextual Recommendations:</h5>
                        <ul style="padding-left: 1.5rem; margin-bottom: 1.5rem; line-height: 1.6;">
                            <li style="margin-bottom: 0.5rem;">{sugg_1}</li>
                            <li>{sugg_2}</li>
                        </ul>
                        <p style="font-size: 0.8rem; color: #6b7280; margin: 0; font-style: italic;">
                            Note: This is an analytical estimation based purely on your calendar timeline. It functions independently of the machine learning clustering model.
                                        '</p>
                    </div>
                    ''', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error parsing the calendar file: {e}")

# ==========================================
# 5. WEARABLE INTEGRATION PROTOTYPE
# ==========================================
with tab5:
    st.header("⌚ Wearable Integration Preview")
    st.info("💡 **Future Prototype / Extension:** This section demonstrates a conceptual pipeline where users could connect a smartwatch (e.g., Apple Health, Google Fit) to analyze live physical metrics against their working profiles.")
    
    with st.container():
        st.subheader("Simulate Live Health Metrics")
        st.markdown("Use the manual sliders below to prototype how the app would process your incoming biometric streams.")
        
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            w_sleep = st.slider("Sleep Last Night (Hours)", 2.0, 12.0, 5.0, 0.5)
            w_hr = st.slider("Avg Daily Heart Rate (bpm)", 40, 150, 85)
            w_sed = st.slider("Continuous Sedentary Time (Minutes)", 0, 300, 150, 15)
            
        with col_w2:
            w_stress = st.slider("Aggregated Stress Level (0-100%)", 0, 100, 75)
            w_steps = st.slider("Steps Taken Today", 0, 20000, 2500, 500)
            w_fatigue = st.slider("Reported Morning Fatigue (1-10)", 1, 10, 8)
            
        st.subheader("⚡ Daily Recovery Assistant")
        st.markdown("Based on your biometrics, here is your real-time physiological readiness to tackle today's calendar:")
        
        # Calculate Recovery Score (0-100)
        sleep_score = max(0, min(100, (w_sleep - 4) * 25))
        stress_score = max(0, min(100, 100 - w_stress))
        fatigue_score = max(0, min(100, (10 - w_fatigue) * 11.1))
        activity_score = max(0, min(100, w_steps / 100))
        
        recovery_score = (sleep_score * 0.4) + (stress_score * 0.25) + (fatigue_score * 0.25) + (activity_score * 0.1)
        
        if recovery_score >= 75:
            readiness = "High"
            r_color = "#10b981"
        elif recovery_score >= 45:
            readiness = "Medium"
            r_color = "#f59e0b"
        else:
            readiness = "Low"
            r_color = "#ef4444"
            
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.metric("Recovery Score", f"{int(recovery_score)} / 100")
            st.markdown(f"**Today's Readiness:** <br><span style='color:{r_color}; font-size:1.5em; font-weight:bold;'>{readiness}</span>", unsafe_allow_html=True)
            
        with col_r2:
            st.markdown("#### 🔄 Daily Adaptation Suggestions")
            if readiness == "High":
                st.success("- Engage in deep, cognitively demanding work early.")
                st.success("- High readiness permits tackling complex friction without rapid cognitive fatigue.")
            elif readiness == "Medium":
                st.warning("- Pace your flow. Break large tasks into smaller 25-minute sprints.")
                st.warning("- Enforce strict start and stop boundaries today to prevent energy trailing.")
            else:
                st.error("- **Prioritize recovery.** Drop non-essential meetings and postpone intense focus blocks.")
                st.error("- Avoid late work completely. End the day early to rebuild your baseline for tomorrow.")

        st.markdown("---")
        st.markdown("**Live Actionable Triggers:**")
        
        recs_given = False
        if w_sleep < 6.0:
            st.warning(f"⚠️ **Sleep Deficit ({w_sleep:.1f}h):** Reduce your meeting load today. Hard boundary: halt complex coding after 15:00.")
            recs_given = True
        if w_stress > 70 or w_hr > 85:
            st.warning(f"⚠️ **Elevated Autonomic Stress:** Step away from all screens immediately for a 15-minute non-cognitive block.")
            recs_given = True
        if w_sed > 90:
            st.info(f"🚶 **Sedentary Warning:** You've been seated for {w_sed} mins. Stand up and move to reset circulation.")
            recs_given = True
            
        if not recs_given:
            st.success("✅ No acute interventions flagged. Biological parameters are steady. It is biologically safe to execute high-stakes cognitive tasks today!")

        st.markdown("---")
        st.caption("🚀 Architecture Note: This prototype utilizes highly deterministic heuristic safety thresholds. A future V2 production deployment would ingest continuous API time-series data to train a lightweight predictive model forecasting true live productivity cliffs.")
