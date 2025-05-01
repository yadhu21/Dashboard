import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier,plot_tree,_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

@st.cache_data 
def load_data():
    data_url = "Data.csv"  
    df = pd.read_csv(data_url)
    return df


st.title("     Sports Data Dashboard")

# Data Loading and Display
data = load_data()
def categorize_performance(goals):
    if goals > 20:
        return "High Scorer"
    elif 10 <= goals <= 20:
        return "Average Scorer"
    else:
        return "Low Scorer"

data["Performance_Category"] = data["Goals"].apply(categorize_performance)

st.sidebar.header("Filter Options")
sport_filter = st.sidebar.multiselect(
    "Select Sport(s):",
    options=data['League'].unique(),
    default=data['League'].unique()
)

filtered_data = data[data['League'].isin(sport_filter)]
st.dataframe(filtered_data)


st.markdown("<div style='height: 5cm;'></div>", unsafe_allow_html=True)
st.subheader("Overall Statistics")
st.write("Choose a metric to visualize")

metric = st.selectbox(
    "Metric:",
    options=["xG","xG Per Avg Match","Shots","OnTarget","Shots Per Avg Match","On Target Per Avg Match","Goals"]
)

fig, ax = plt.subplots()
st.write(f"The Top Goal Scorer of League {sport_filter} (above 21 goals)",divider="blue")
sns.barplot(data=filtered_data[filtered_data[metric]>filtered_data[metric].quantile(0.95)], x="Player Names", y=metric, ax=ax,palette="cool",hue="Player Names")
plt.xticks(rotation=75)
st.pyplot(fig)






st.markdown("<div style='height: 5cm;'></div>", unsafe_allow_html=True)


st.header("Descriptive Statistics:")
st.write(filtered_data.describe())


correlation_matrix = filtered_data[["Goals", "xG", "Shots", "OnTarget"]].corr()
st.header("\nCorrelation Matrix:")
st.write(correlation_matrix)
st.markdown("<div style='height: 5cm;'></div>", unsafe_allow_html=True)

st.title("Football Player Performance Analysis")


st.subheader("Goals vs Expected Goals (xG)")

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=filtered_data, x="xG", y="Goals", size="Shots", hue="OnTarget", palette="viridis", ax=ax)
ax.set_title("Goals vs Expected Goals (xG)")
ax.set_xlabel("Expected Goals (xG)")
ax.set_ylabel("Goals")
st.pyplot(fig)
st.markdown("<div style='height: 5cm;'></div>", unsafe_allow_html=True)


st.subheader("League Count in Each Country")
league_count = data.groupby('Country')['League'].nunique().reset_index()
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=league_count, x="Country", y="League", ax=ax, palette="coolwarm",hue="Country")
ax.set_title("Number of Leagues in Each Country")
ax.set_xlabel("Country")
ax.set_ylabel("Number of Leagues")
st.pyplot(fig)




st.markdown("<div style='height: 5cm;'></div>", unsafe_allow_html=True)

st.header("Linear Regression For Predicting the Goal using Machine Learning")

X = filtered_data[['xG', 'Shots', 'Mins']]


y = filtered_data['Goals']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




model = LinearRegression()
model.fit(X_train, y_train)





y_pred = model.predict(X_test)


fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, label="Predicted vs Actual", color="blue")



min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
ax.plot([min_val, max_val], [min_val, max_val], "red", linestyle="dotted", label="Perfect Prediction Line")
st.pyplot(fig)



st.write("Coefficients:", model.coef_)
st.write("Intercept:", model.intercept_)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("Mean Squared Error:", mse)
st.write("R² Score:", r2)

st.markdown("<div style='height: 5cm;'></div>", unsafe_allow_html=True)

st.title('Football Goal Prediction using Linear Regression')




xG = st.slider('xG', 0, 30, 15)
shots = st.slider('Shots', 0, 200, 5)
mins = st.slider(' Mins', 5, 4000, 100)

user_input = pd.DataFrame([[xG,shots,mins]], columns=['xG', 'Shots', 'Mins'])


predicted_goals = model.predict(user_input)


st.write(f'Predicted number of goals: {predicted_goals[0]:.2f}')







st.markdown("<div style='height: 5cm;'></div>", unsafe_allow_html=True)


st.header("Decision Tree  (Players Goal scoring division based on Performance factors )")





features = ['xG', 'Shots', 'Mins', 'OnTarget', 'Matches_Played']
X = filtered_data[features]
y = filtered_data["Performance_Category"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                                   
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf.fit(X_train, y_train)
zoom_level = st.slider('Zoom Level (Adjust the size of the decision tree plot)', min_value=5, max_value=20, value=12, step=1)

fig, ax = plt.subplots(figsize=(zoom_level, zoom_level * 0.75))
plot_tree(clf, feature_names=features, class_names=clf.classes_, filled=True, ax=ax)         

plt.title("Decision Tree for Player Performance Classification")

st.pyplot(fig)

y_pred = clf.predict(X_test)

st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

st.write("\nAccuracy Score:", accuracy_score(y_test, y_pred))





def predict_and_suggest_improvements(xG, on_target, shots, mins, matches_played):
    
    user_data = pd.DataFrame({
        'xG': [xG],
        'Shots': [shots],
        'Mins': [mins],
        'OnTarget': [on_target],
        'Matches_Played': [matches_played]
    })
    

    prediction = clf.predict(user_data)[0]
   
    suggestions=[]
    

    if str(prediction) == "Low Scorer":
        
        if xG <= 07.66:
            suggestions.append("Increase your expected goals (xG) by improving shot quality and positioning.")
        if on_target <= 16.5:
            suggestions.append("Focus on increasing shots on target through accuracy drills.")
        if shots <= 82.0:
            suggestions.append("Attempt more shots per match to create better scoring opportunities.")
        if mins < 900:
            suggestions.append("Increase playtime to gain experience and consistency.")
    
    elif str(prediction) == "Average Scorer":
        if xG <= 17.225:
            suggestions.append("Work on enhancing shot precision to convert more chances and increase xG.")
        if on_target <= 49.5:
            suggestions.append("Focus on training to improve shot accuracy and aiming for tighter targets.")
        if shots <= 150:
            suggestions.append("Shoot more consistently in matches to capitalize on opportunities.")
        if mins < 1800:
            suggestions.append("Focus on fitness and stamina to sustain high performance throughout matches.")

    elif str(prediction) == "Good Scorer":
        suggestions.append("Maintain your current form and focus on team collaboration to achieve excellence.")
        suggestions.append("Work on minor optimizations like decision-making under pressure.")
    
    return prediction,suggestions


st.title("Player Performance Prediction and Improvement Suggestions")

xG = st.number_input("Enter Expected Goals (xG):", min_value=0.0, step=0.1)
on_target = st.number_input("Enter Shots On Target:", min_value=0, step=1)
shots = st.number_input("Enter Total Shots:", min_value=0, step=1)
mins = st.number_input("Enter Total Minutes Played:", min_value=0, step=1)
matches_played = st.number_input("Enter Total Matches Played:", min_value=0, step=1)

if st.button("Predict and Suggest Improvements"):
    performance, improvements = predict_and_suggest_improvements(xG, on_target, shots, mins, matches_played)
    st.success(f"The player's performance is predicted to be: **{performance}**")
    
    if improvements:
        st.info("Here are suggestions to improve performance:")
        for suggestion in improvements:
            st.write(f"- {suggestion}")
            
    
    




st.markdown("<div style='height: 5cm;'></div>", unsafe_allow_html=True)


st.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_data.to_csv(index=False),
    file_name="filtered_sports_data.csv",
    mime="text/csv",
)



st.markdown("Built with Streamlit by Yadhu Krishnan C K and team")
