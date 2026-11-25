"""Thermostat primer for BTP report v2, section 3. Room starts 10C, target 22C, leak 0.1*(T-10) per step, heater power >= 0.
P: kp=.5 ; PI: +ki=.2 ; PID: +kd=.15."""
def sim(kp,ki,kd,n=80):
    T=10.;I=0.;ep=None;out=[]
    for t in range(1,n+1):
        e=22-T; I+=e; D=0. if ep is None else (e-ep); ep=e
        P_=kp*e; I_=ki*I; D_=kd*D; pw=max(0,P_+I_+D_)
        out.append((t,T,e,P_,I_,D_,pw)); T=T+pw-0.1*(T-10)
    return out
if __name__=="__main__":
    for nm,(kp,ki,kd) in {"P":(.5,0,0),"PI":(.5,.2,0),"PID":(.5,.2,.15)}.items():
        o=sim(kp,ki,kd); print(nm,"max T %.2f"%max(r[1] for r in o))
        for r in o[:6]+[o[9],o[19],o[79]]: print("  t=%2d T=%5.2f e=%5.2f P=%5.2f I=%5.2f D=%5.2f power=%5.2f"%r)
