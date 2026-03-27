
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc,confusion_matrix

st.set_page_config(layout="wide")
st.title("LPG Distinction-Level Analytics Dashboard")

df=pd.read_csv("distinction_dataset.csv")

# KPIs
col1,col2,col3,col4=st.columns(4)
col1.metric("Total Customers",len(df))
col2.metric("Avg Delivery Days",round(df["Delivery_Days"].mean(),2))
col3.metric("Complaint Rate",round(df["Complaint"].mean()*100,2))
col4.metric("Avg Satisfaction",round(df["Overall_Satisfaction"].mean(),2))

st.header("Descriptive Analysis")
st.plotly_chart(px.histogram(df,x="Booking_Method",color="Complaint",title="Booking vs Complaint"))
st.plotly_chart(px.box(df,x="City_Tier",y="Delivery_Days",title="City vs Delivery"))
st.plotly_chart(px.scatter(df,x="Delivery_Days",y="Overall_Satisfaction",color="Complaint",title="Delivery vs Satisfaction"))

st.header("Diagnostic Insights")
st.write("Longer delivery times and non-digital booking methods lead to higher complaints.")

# Encoding
df_enc=df.copy()
le=LabelEncoder()
for col in df_enc.columns:
    if df_enc[col].dtype=="object":
        df_enc[col]=le.fit_transform(df_enc[col])

X=df_enc.drop("Complaint",axis=1)
y=df_enc["Complaint"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=RandomForestClassifier()
model.fit(X_train,y_train)

pred=model.predict(X_test)
prob=model.predict_proba(X_test)[:,1]

st.header("Predictive Model")
st.write("Accuracy",accuracy_score(y_test,pred))
st.write("Precision",precision_score(y_test,pred))
st.write("Recall",recall_score(y_test,pred))
st.write("F1",f1_score(y_test,pred))

fpr,tpr,_=roc_curve(y_test,prob)
fig=go.Figure()
fig.add_trace(go.Scatter(x=fpr,y=tpr,name="ROC"))
fig.add_trace(go.Scatter(x=[0,1],y=[0,1],name="Baseline"))
st.plotly_chart(fig)

cm=confusion_matrix(y_test,pred)
st.write("Confusion Matrix",cm)

# Clustering
st.header("Clustering")
km=KMeans(n_clusters=4,n_init=10)
df["Cluster"]=km.fit_predict(X)
st.plotly_chart(px.scatter(df,x="Delivery_Days",y="Overall_Satisfaction",color=df["Cluster"].astype(str)))

# Association
st.header("Association")
conf=df[df["Booking_Method"]=="Agent"]["Complaint"].mean()
lift=conf/df["Complaint"].mean()
st.write("Confidence",conf)
st.write("Lift",lift)

st.header("Prescriptive")
st.write("Promote app usage, optimize delivery routes, and target high-risk customers.")
