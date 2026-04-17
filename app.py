import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
def get_clustering_model(data, features, n_clusters=3):
    """
    Fits and returns the Scaler and KMeans model without mutating global variables.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)
    
    return scaler, kmeans

def generate_cluster_profiles(scaler, kmeans, features, n_clusters):
    """
    Analyzes centroids and returns human-readable cluster profiles and the centroid DataFrame.
    """
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
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
        
    return cluster_profiles, centroid_df


# ==========================================
# 3. UI TABS & HIGH-LEVEL STRUCTURE
# ==========================================
st.title("🧘 Work-From-Home Burnout & Suggestions App")
st.markdown("Analyze WFH habits, discover different *person types*, and get tailored recovery suggestions.")

tab1, tab2, tab3 = st.tabs(["🗂️ Data Explorer", "🔍 Explore Clusters", "👤 Client Matcher"])

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
    
    # 1. Fetch cached model
    scaler_temp, kmeans_temp = get_clustering_model(df, cluster_features, n_clusters)
    profiles_temp, centroids_temp = generate_cluster_profiles(scaler_temp, kmeans_temp, cluster_features, n_clusters)
    
    # 2. Create explicit local copy to mutate and visualize
    temp_df = df.copy()
    temp_df['Cluster'] = kmeans_temp.labels_  # Safe assignment on local copy
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
    final_scaler, final_kmeans = get_clustering_model(df, cluster_features, final_k)
    final_profiles, _ = generate_cluster_profiles(final_scaler, final_kmeans, cluster_features, final_k)
    
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
