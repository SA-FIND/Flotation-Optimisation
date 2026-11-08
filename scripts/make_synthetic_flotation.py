import numpy as np, pandas as pd, sys
# AR(1) targets + a real sensor signal (Amina Flow hourly level drives grade), plus noise sensors
rng=np.random.default_rng(1); H=3000; per=60; n=H*per
amina_h=rng.normal(0,1,H)
fe=np.empty(H); si=np.empty(H); fe[0]=65; si[0]=2.3
for t in range(1,H):
    fe[t]=65+0.6*(fe[t-1]-65)+0.6*amina_h[t]+rng.normal(0,0.8)
    si[t]=2.3+0.6*(si[t-1]-2.3)-0.5*amina_h[t]+rng.normal(0,0.7)
df=pd.DataFrame({'date':pd.date_range('2017-03-10',periods=n,freq='1min').strftime('%Y-%m-%d %H:%M:%S')})
df['% Iron Feed']=np.repeat(56+rng.normal(0,1,H//24+1),24*per)[:n]
df['% Silica Feed']=np.repeat(14+rng.normal(0,1,H//24+1),24*per)[:n]
df['Amina Flow']=np.repeat(500+50*amina_h,per)+rng.normal(0,10,n)
for c in ['Starch Flow','Ore Pulp Flow','Ore Pulp pH','Ore Pulp Density']: df[c]=rng.normal(100,10,n)
df['% Iron Concentrate']=np.repeat(fe,per); df['% Silica Concentrate']=np.repeat(si,per)
for c in df.columns[1:]: df[c]=df[c].round(4).astype(str).str.replace('.',',',regex=False)
df.to_csv(sys.argv[1],index=False)
