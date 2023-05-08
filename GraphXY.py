import matplotlib.pyplot as plt
from datetime import datetime 
import numpy as np

#def plot_graph(FLOW_SF, X, time_data, dni, Tout, Tin, model):
def plot_graph(DATA,  X, model):

    
    FLOW_SF = DATA['Flow'].astype(float).values.reshape(-1, 1)  


    dni = np.array(DATA['DNIreal']).reshape(-1, 1) 
    Tout = np.array(DATA['TOUT']).reshape(-1, 1) 
    Tin = np.array(DATA['TIN']).reshape(-1, 1) 


    fig, ax1 = plt.subplots(figsize=(16, 6))

    #y_pred = model.predict(X, verbose=0)
    #y_pred = np.clip(y_pred, a_min=1800, a_max=None)

    ax1.plot(DATA['Time'][:len(X)], FLOW_SF[:len(X)], color='blue', linewidth=3)
    ax1.plot(DATA['Time'][:len(X)], model.predict(X), color='red', linewidth=3) 

    ax2 = ax1.twinx()
    ax2.plot(DATA['Time'][:len(X)], dni[:len(X)], color='yellow', linewidth=2)
    ax2.set_ylabel('DNI') 
    ax2.set_ylim(0, 1000)  

    ax3 = ax1.twinx()
    ax3.plot(DATA['Time'][:len(X)], Tout[:len(X)], color='black', linewidth=1) 
    ax3.spines['right'].set_position(('axes', 1.1))
    ax3.set_ylabel('Tout') 
    ax3.yaxis.set_label_coords(1.08, 0.5)   
    ax3.set_ylim(100, 400)  

    ax4 = ax1.twinx()
    ax4.plot(DATA['Time'][:len(X)], Tin[:len(X)], color='gray', linewidth=1) 
    ax4.spines['right'].set_position(('axes', 1.1))
    ax4.set_ylabel('Tin') 
    ax4.yaxis.set_label_coords(1.1, 0.5)   
    ax4.set_ylim(100, 400)

    ax1.set_title('Caudal real vs Modelo')
    ax1.set_xlabel('tiempo')
    ax1.set_ylabel('caudal')
    ax1.set_xticks(DATA['Time'][::30])

 
    # Agregar cursor
    time_float = []
    for t in DATA['Time']:
        time_obj = datetime.strptime(t, '%H:%M')
        time_float.append(time_obj.hour * 3600 + time_obj.minute * 60)

    cursor = ax1.axvline(x=0, color='black', linewidth=1, linestyle='--')
    textpos = ax1.text(0.01, 0.7, '', transform=ax1.transAxes, ha='left')

    def validateX(x):
        x = x if x is not None else None
        x = max(0, min(x, len(time_float)-1)) if x is not None else None
        return x


    def on_move(event):
        x, y = event.xdata, event.ydata
        x = validateX(x)
    
        if validateX(x) == None :
            return 

        cursor.set_xdata([x]) 
    
        index = round(x)     
        y1 = FLOW_SF[index]
        y2 = model.predict(X)[index]
        y3 = dni[index]
        y4 = Tin[index]  
        y5 = Tout[index]  

        qdif = y1 - y2
        dt = y5 - y4
    
        textpos.set_text('Hora: ' + DATA['Time'][index] + 
                     '\nCaudal real: ' + str(y1.round(2)) + 
                     '\nCaudal modelo: ' + str(y2.round(2)) + 
                     '\nCaudal dif(r-m): ' + str(qdif.round(2)) + 
                     '\nDNI: ' + str(y3.round(2)) + 
                     '\nTin: ' + str(y4.round(2))+ 
                     '\nTout: ' + str(y5.round(2))+ 
                     '\nDT: ' + str(dt.round(2)) )
       

        fig.canvas.draw_idle()


    fig.canvas.mpl_connect('motion_notify_event', on_move) 

    plt.show()

 