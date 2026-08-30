import io, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title='PaddyYield Intelligence', page_icon='🌾', layout='wide', initial_sidebar_state='expanded')
ROOT=Path(__file__).parent
B=joblib.load(ROOT/'model_bundle.joblib')
DF=pd.read_csv(ROOT/'paddydataset.csv')
DF.columns=DF.columns.str.strip()
LABELS=B['labels']; MODELS=B['models']; FEATURES=B['features']; NUM=B['numeric']; CAT=B['categorical']; TARGET=B['target']
RANGES={'Low: 0-10,000':'0–10,000 kg','Moderate: 10,001-20,000':'10,001–20,000 kg','High: 20,001-30,000':'20,001–30,000 kg','Very High: 30,001-40,000':'30,001–40,000 kg'}
SHORT={x:x.split(':')[0] for x in LABELS}

st.markdown('''<style>
.block-container{padding-top:1.2rem;max-width:1500px}.hero{padding:26px;border:1px solid rgba(128,128,128,.22);border-radius:22px;background:linear-gradient(135deg,rgba(46,125,50,.12),rgba(255,193,7,.08));margin-bottom:18px}.hero h1{margin:0;font-size:2.35rem}.hero p{margin:.5rem 0 0;opacity:.78}.card{padding:18px;border:1px solid rgba(128,128,128,.20);border-radius:16px}.small{font-size:.86rem;opacity:.72}.stTabs [data-baseweb="tab-list"]{gap:8px}.stTabs [data-baseweb="tab"]{padding:10px 14px}
</style>''',unsafe_allow_html=True)

@st.cache_data
def category_counts(): return DF['Paddy Yield Category'].value_counts().reindex(LABELS).fillna(0)

if 'history' not in st.session_state: st.session_state.history=[]

st.sidebar.markdown('## 🌾 PaddyYield Intelligence')
st.sidebar.caption('4-model machine-learning decision platform')
page=st.sidebar.radio('Navigate', ['Executive Dashboard','AI Prediction','Model Arena','What-if Simulator','Explainability','Data Explorer','Prediction History','Project Info'])
st.sidebar.divider()
st.sidebar.caption(f"Dataset: {len(DF):,} records after notebook-style duplicate cleaning")
st.sidebar.caption(f"Predictors: {len(FEATURES)}")

if page=='Executive Dashboard':
 st.markdown('<div class="hero"><h1>🌾 Paddy Yield Intelligence</h1><p>Advanced multiclass prediction and model-evaluation dashboard based on the supplied paddy-yield modelling workflow.</p></div>',unsafe_allow_html=True)
 m=B['metrics']; best=max(m,key=lambda x:x['Macro F1']); counts=category_counts()
 c=st.columns(6); c[0].metric('Records',f"{len(DF):,}"); c[1].metric('Predictors',len(FEATURES)); c[2].metric('Yield mean',f"{DF[TARGET].mean():,.0f} kg"); c[3].metric('Yield median',f"{DF[TARGET].median():,.0f} kg"); c[4].metric('Best Macro F1',f"{best['Macro F1']:.2%}"); c[5].metric('Best model',best['Model'])
 a,b=st.columns([1,1]);
 with a:
  fig=px.bar(x=[SHORT[x] for x in LABELS],y=counts.values,text=counts.values,title='Yield Category Distribution',labels={'x':'Category','y':'Records'}); fig.update_layout(height=400); st.plotly_chart(fig,use_container_width=True)
 with b:
  fig=px.histogram(DF,x=TARGET,color='Paddy Yield Category',nbins=35,title='Observed Paddy Yield Distribution'); fig.update_layout(height=400); st.plotly_chart(fig,use_container_width=True)
 st.subheader('Four-model leaderboard')
 lm=pd.DataFrame(m).sort_values('Macro F1',ascending=False).reset_index(drop=True); lm.insert(0,'Rank',np.arange(1,len(lm)+1)); st.dataframe(lm.style.format({x:'{:.2%}' for x in ['Accuracy','Macro Precision','Macro Recall','Macro F1','ROC-AUC']}),use_container_width=True,hide_index=True)
 st.info('The notebook defines four target classes and uses Macro F1 prominently because it gives each yield category equal weight. The application keeps the four modelling tracks visible instead of hiding them behind a single model.')

elif page=='AI Prediction':
 st.title('🤖 AI Prediction Studio'); st.caption('Run one model, compare all four models, or use a soft-voting ensemble.')
 mode=st.radio('Prediction mode',['Compare all 4 models','Single model','Ensemble (mean probability)'],horizontal=True)
 selected=st.selectbox('Model',list(MODELS),disabled=(mode!='Single model'))
 # seed from median/mode values
 values={}
 with st.form('predict'):
  tabs=st.tabs(['🌱 Farm & Crop','🌧 Rainfall & AI','🌡 Temperature','💨 Wind & Humidity','🧪 Field Inputs'])
  groups=[ [c for c in CAT if c in FEATURES], [c for c in NUM if 'Rain' in c or 'AI' in c], [c for c in NUM if 'temp' in c], [c for c in NUM if 'Wind' in c or 'Humidity' in c], [c for c in NUM if c not in [z for g in [[c for c in NUM if 'Rain' in c or 'AI' in c],[c for c in NUM if 'temp' in c],[c for c in NUM if 'Wind' in c or 'Humidity' in c]] for z in g]] ]
  for tab,cols in zip(tabs,groups):
   with tab:
    cc=st.columns(3)
    for i,col in enumerate(cols):
     with cc[i%3]:
      if col in CAT:
       opts=sorted(DF[col].dropna().astype(str).unique()); values[col]=st.selectbox(col,opts,index=0,key='p_'+col)
      else:
       s=pd.to_numeric(DF[col],errors='coerce').dropna(); lo=float(s.quantile(.01)); hi=float(s.quantile(.99)); med=float(s.median()); values[col]=st.number_input(col,min_value=lo,max_value=hi,value=med,key='p_'+col)
  go=st.form_submit_button('🚀 Generate Prediction',use_container_width=True,type='primary')
 if go:
  row=pd.DataFrame([values],columns=FEATURES)
  probs={}; preds={}
  for name,model in MODELS.items():
   raw=model.predict_proba(row)[0]
   if name=='ANN':
    classes=B['ann_label_classes']; arr=np.zeros(len(LABELS));
    for i,cl in enumerate(classes): arr[LABELS.index(cl)]=raw[i]
    raw=arr; pred=LABELS[int(np.argmax(raw))]
   else: pred=model.predict(row)[0]
   probs[name]=raw; preds[name]=pred
  if mode=='Single model': avg=probs[selected]; final=preds[selected]
  else: avg=np.mean(list(probs.values()),axis=0); final=LABELS[int(np.argmax(avg))]
  conf=float(np.max(avg))
  st.divider(); a,b,c=st.columns(3); a.metric('Predicted Yield Category',SHORT[final]); b.metric('Confidence',f'{conf:.1%}'); c.metric('Expected Range',RANGES[final])
  if mode=='Compare all 4 models':
   table=[]
   for n in MODELS: table.append({'Model':n,'Prediction':SHORT[preds[n]],'Confidence':float(max(probs[n]))})
   st.dataframe(pd.DataFrame(table).style.format({'Confidence':'{:.1%}'}),use_container_width=True,hide_index=True)
  ptable=pd.DataFrame({'Category':[SHORT[x] for x in LABELS],'Probability':avg}); fig=px.bar(ptable,x='Category',y='Probability',range_y=[0,1],text=ptable.Probability.map(lambda x:f'{x:.1%}'),title='Prediction probability profile'); fig.update_traces(textposition='outside'); st.plotly_chart(fig,use_container_width=True)
  if st.button('Save this prediction to history'): st.session_state.history.append({'Timestamp':pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),'Mode':mode,'Model':selected if mode=='Single model' else mode,'Prediction':SHORT[final],'Confidence':conf})
  st.warning('Confidence is the highest model probability; it is not a guarantee of actual farm yield.')

elif page=='Model Arena':
 st.title('🏆 Model Arena'); st.caption('Side-by-side evaluation of Logistic Regression, ANN, Random Forest and KNN.')
 metrics=pd.DataFrame(B['metrics']).sort_values('Macro F1',ascending=False); metric=st.selectbox('Leaderboard metric',['Accuracy','Macro Precision','Macro Recall','Macro F1','ROC-AUC'])
 fig=px.bar(metrics.sort_values(metric),x=metric,y='Model',orientation='h',range_x=[0,1],text_auto='.2%',title=f'Model ranking by {metric}'); st.plotly_chart(fig,use_container_width=True)
 st.dataframe(metrics.style.format({x:'{:.2%}' for x in ['Accuracy','Macro Precision','Macro Recall','Macro F1','ROC-AUC']}),use_container_width=True,hide_index=True)
 tabs=st.tabs(list(MODELS))
 for tab,name in zip(tabs,MODELS):
  with tab:
   r=next(x for x in B['metrics'] if x['Model']==name); a,b,c=st.columns(3); a.metric('Accuracy',f"{r['Accuracy']:.2%}"); b.metric('Macro F1',f"{r['Macro F1']:.2%}"); c.metric('ROC-AUC',f"{r['ROC-AUC']:.2%}")
   x,y=st.columns(2)
   with x:
    cm=np.array(B['confusion'][name]); fig=px.imshow(cm,x=[SHORT[x] for x in LABELS],y=[SHORT[x] for x in LABELS],text_auto=True,labels={'x':'Predicted','y':'Actual','color':'Count'},title='Confusion Matrix'); st.plotly_chart(fig,use_container_width=True)
   with y:
    fig=go.Figure();
    for lab,d in B['roc'][name].items(): fig.add_trace(go.Scatter(x=d['fpr'],y=d['tpr'],mode='lines',name=f'{SHORT[lab]} ({d["auc"]:.3f})'))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Random',line={'dash':'dash'})); fig.update_layout(title='Multiclass ROC curves',xaxis_title='False Positive Rate',yaxis_title='True Positive Rate',height=430); st.plotly_chart(fig,use_container_width=True)
   fig=go.Figure()
   for lab,d in B['pr'][name].items(): fig.add_trace(go.Scatter(x=d['recall'],y=d['precision'],mode='lines',name=f'{SHORT[lab]} (AP {d["ap"]:.3f})'))
   fig.update_layout(title='Multiclass Precision–Recall curves',xaxis_title='Recall',yaxis_title='Precision',xaxis_range=[0,1],yaxis_range=[0,1],height=430); st.plotly_chart(fig,use_container_width=True)
   st.markdown('**Best hyperparameters used in the application**'); st.json(B['best_params'][name])

elif page=='What-if Simulator':
 st.title('🧪 What-if Scenario Simulator'); st.caption('Start from an actual dataset record, modify conditions, and observe probability shifts across all four models.')
 idx=st.slider('Base record',0,len(DF)-1,0); base=DF.loc[[idx],FEATURES].copy(); st.write('Base observed yield:',f"**{DF.loc[idx,TARGET]:,.0f} kg** — {SHORT[DF.loc[idx,'Paddy Yield Category']]}")
 mods=st.multiselect('Variables to change',NUM,default=NUM[:3]); scenario=base.copy(); cc=st.columns(2)
 for i,col in enumerate(mods):
  with cc[i%2]: scenario.loc[scenario.index[0],col]=st.number_input(col,min_value=float(DF[col].min()),max_value=float(DF[col].max()),value=float(base.iloc[0][col]),key='w_'+col)
 if st.button('Run scenario',type='primary'):
  rows=[]
  for name,m in MODELS.items():
   p0=m.predict_proba(base)[0]; p1=m.predict_proba(scenario)[0]
   if name=='ANN':
    p0=np.array([p0[B['ann_label_classes'].index(l)] for l in LABELS]); p1=np.array([p1[B['ann_label_classes'].index(l)] for l in LABELS])
   rows.append(pd.DataFrame({'Category':[SHORT[x] for x in LABELS],'Probability':p0,'Model':name,'Scenario':'Base'})); rows.append(pd.DataFrame({'Category':[SHORT[x] for x in LABELS],'Probability':p1,'Model':name,'Scenario':'What-if'}))
  chart=pd.concat(rows); fig=px.bar(chart,x='Category',y='Probability',color='Scenario',facet_col='Model',barmode='group',range_y=[0,1],title='Probability shift across the four models'); fig.update_layout(height=600); st.plotly_chart(fig,use_container_width=True)
  changes=pd.DataFrame({'Variable':mods,'Base':[base.iloc[0][c] for c in mods],'Scenario':[scenario.iloc[0][c] for c in mods]}); st.dataframe(changes,use_container_width=True,hide_index=True)

elif page=='Explainability':
 st.title('🔍 Explainability & Model Insight'); st.caption('Model-specific interpretation views plus a common permutation-importance comparison.')
 name=st.selectbox('Model',list(MODELS))
 if name=='Logistic Regression':
  st.subheader('Logistic Regression — Coefficient View')
  m=MODELS[name]
  try:
   prep=m.named_steps['preprocessor']; clf=m.named_steps['model']; feat_names=prep.get_feature_names_out(); coef=np.mean(np.abs(clf.coef_),axis=0)
   fi=pd.DataFrame({'Feature':feat_names,'Absolute coefficient':coef}).sort_values('Absolute coefficient',ascending=False).head(20).sort_values('Absolute coefficient')
   fig=px.bar(fi,x='Absolute coefficient',y='Feature',orientation='h',title='Top 20 absolute Logistic Regression coefficients'); st.plotly_chart(fig,use_container_width=True)
  except Exception: st.info('Coefficient details are unavailable for this stored model.')
 elif name=='Random Forest':
   st.subheader('Random Forest — Feature Importance')
   fi=pd.DataFrame({'Feature':B['importance'][name]['feature'],'Importance':B['importance'][name]['mean'],'Std':B['importance'][name]['std']}).sort_values('Importance',ascending=False).head(20).sort_values('Importance')
   fig=px.bar(fi,x='Importance',y='Feature',orientation='h',error_x='Std',title='Top 20 Random Forest permutation importance'); st.plotly_chart(fig,use_container_width=True)
 elif name=='KNN':
   st.subheader('KNN — Permutation Importance')
   fi=pd.DataFrame({'Feature':B['importance'][name]['feature'],'Importance':B['importance'][name]['mean'],'Std':B['importance'][name]['std']}).sort_values('Importance',ascending=False).head(20).sort_values('Importance')
   fig=px.bar(fi,x='Importance',y='Feature',orientation='h',error_x='Std',title='Top 20 KNN permutation importance'); st.plotly_chart(fig,use_container_width=True)
 else:
   st.subheader('ANN — Neural Network Analysis')
   st.write('Architecture / tuning configuration:')
   st.json(B['best_params'][name])
   fi=pd.DataFrame({'Feature':B['importance'][name]['feature'],'Importance':B['importance'][name]['mean'],'Std':B['importance'][name]['std']}).sort_values('Importance',ascending=False).head(20).sort_values('Importance')
   fig=px.bar(fi,x='Importance',y='Feature',orientation='h',error_x='Std',title='ANN permutation importance'); st.plotly_chart(fig,use_container_width=True)
 st.divider()
 st.subheader('Learning Curve')
 lc=B['learning'][name]; fig=go.Figure(); fig.add_trace(go.Scatter(x=lc['sizes'],y=lc['train_mean'],mode='lines+markers',name='Training Macro F1')); fig.add_trace(go.Scatter(x=lc['sizes'],y=lc['val_mean'],mode='lines+markers',name='Validation Macro F1')); fig.update_layout(title=f'{name} Learning Curve',xaxis_title='Training Records',yaxis_title='Macro F1',yaxis_range=[0,1]); st.plotly_chart(fig,use_container_width=True)

elif page=='Data Explorer':
 st.title('📊 Data Explorer'); st.caption('Explore the cleaned modelling dataset and the four yield categories.')
 a,b,c=st.columns(3); catf=a.selectbox('Yield category',['All']+LABELS); x=b.selectbox('X variable',NUM); y=c.selectbox('Y variable',NUM+[TARGET],index=len(NUM))
 view=DF if catf=='All' else DF[DF['Paddy Yield Category']==catf]; sample=view.sample(min(len(view),1200),random_state=42)
 fig=px.scatter(sample,x=x,y=y,color='Paddy Yield Category',hover_data=[z for z in ['Agriblock','Variety','Soil Types',TARGET] if z in DF.columns],title=f'{x} vs {y}'); fig.update_layout(height=560); st.plotly_chart(fig,use_container_width=True)
 st.subheader('Category profile'); st.dataframe(DF.groupby('Paddy Yield Category')[NUM].mean().reindex(LABELS).round(2),use_container_width=True)
 st.subheader('Records'); st.dataframe(view.head(500),use_container_width=True,hide_index=True)

elif page=='Prediction History':
 st.title('📋 Prediction History'); st.caption('Session-level prediction audit trail. It resets when the Streamlit session ends.')
 if st.session_state.history:
  h=pd.DataFrame(st.session_state.history); st.dataframe(h,use_container_width=True,hide_index=True); st.download_button('Download history CSV',h.to_csv(index=False).encode(),file_name='prediction_history.csv',mime='text/csv')
 else: st.info('No predictions saved yet. Run a prediction and choose “Save this prediction to history”.')

else:
 st.title('ℹ️ Project Architecture')
 st.markdown('''### What is inside this advanced version?
**Data layer:** supplied `paddydataset.csv`, duplicate cleaning, missing-value handling, categorical one-hot encoding and numerical standardisation.

**Four modelling tracks:** Logistic Regression, ANN (MLPClassifier), Random Forest and KNN. The modelling choices follow the uploaded notebook's terminology and target classes.

**Evaluation layer:** Accuracy, Macro Precision, Macro Recall, Macro F1, ROC-AUC, Log Loss, confusion matrices, multiclass ROC curves, precision–recall curves, 5-fold CV summaries, learning curves and model-specific explainability.

**Decision layer:** single-model prediction, four-model comparison, soft-voting probability ensemble, what-if simulation and session prediction history.

**Deployment:** GitHub + Streamlit Community Cloud ready. The trained models are stored in `model_bundle.joblib`, so Streamlit does not need to retrain during every page load.''')
 st.subheader('Target definition'); st.table(pd.DataFrame({'Class':[SHORT[x] for x in LABELS],'Range':[RANGES[x] for x in LABELS]}))
 st.warning('This is an academic/analytical decision-support application. Predictions should not be interpreted as guaranteed agricultural outcomes.')
