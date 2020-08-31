from Assignment1 import *   

def test_case():
    '''df = pd.read_csv('Test.csv')
    #print(df)
    print('Dataset entropy : ',get_entropy_of_dataset(df))  #0.9709505944546686
    print('Sky entropy : ',get_entropy_of_attribute(df, 'Sky')) #0.9509775004326937
    print('Sky IG : ', get_information_gain(df, 'Sky')) #0.01997309402197489
    print('Airtemp entropy : ',get_entropy_of_attribute(df, 'Airtemp')) #0.6490224995673063
    print('Airtemp IG : ', get_information_gain(df, 'Airtemp')) #0.3219280948873623
    print('Humidity entropy : ',get_entropy_of_attribute(df, 'Humidity')) #0.9509775004326937
    print('Humidity IG : ', get_information_gain(df, 'Humidity')) #0.01997309402197489
    print('Water entropy : ',get_entropy_of_attribute(df, 'Water')) #0.8
    print('Water IG : ', get_information_gain(df, 'Water')) #0.17095059445466854
    print('Forecast entropy : ',get_entropy_of_attribute(df, 'Forecast')) #0.9509775004326937
    print('Forecast IG : ', get_information_gain(df, 'Forecast')) #0.01997309402197489
    print(get_selected_attribute(df)) #Airtemp'''
    
    
    '''df = pd.read_csv('Test1.csv')
    print('Dataset entropy : ', get_entropy_of_dataset(df))  #0.9402859586706311
    print('Age entropy : ', get_entropy_of_attribute(df, 'Age')) #0.6324823551623816
    print('Age IG : ', get_information_gain(df, 'Age')) #0.30780360350824953
    print('Income entropy : ', get_entropy_of_attribute(df, 'Income')) #0.9110633930116763
    print('Income IG : ', get_information_gain(df, 'Income')) #0.02922256565895487
    print('Student entropy : ', get_entropy_of_attribute(df, 'Student')) #0.7884504573082896
    print('Student IG : ', get_information_gain(df, 'Student')) #0.15183550136234159
    print('Credit_rating entropy : ', get_entropy_of_attribute(df, 'Credit_rating')) #0.8921589282623617
    print('Credit_rating IG : ', get_information_gain(df, 'Credit_rating')) #0.04812703040826949
    print(get_selected_attribute(df)) #Age'''
    
    
    
    '''df = pd.read_csv('Test2.csv')
    print('Dataset entropy : ', get_entropy_of_dataset(df)) #0.9852281360342515
    print('Salary entropy : ', get_entropy_of_attribute(df,'salary')) #0.5156629249195446
    print('Salary IG : ', get_information_gain(df,'salary')) #0.46956521111470695
    print('Location entropy : ', get_entropy_of_attribute(df,'location')) #0.2857142857142857
    print('Location IG : ', get_information_gain(df,'location')) #0.6995138503199658
    print(get_selected_attribute(df))'''
    


    '''df = pd.read_csv('Test3.csv')
    print('Dataset entropy : ', get_entropy_of_dataset(df)) #0.9709505944546686
    print('Toothed entropy : ', get_entropy_of_attribute(df,'toothed')) #0.963547202339972
    print('Toothed IG : ', get_information_gain(df,'toothed')) #0.007403392114696539
    print('Breathes entropy : ', get_entropy_of_attribute(df,'breathes')) #0.8264662506490407
    print('Breathes IG : ', get_information_gain(df,'breathes')) #0.1444843438056279
    print('Legs entropy : ', get_entropy_of_attribute(df,'legs')) #0.4141709450076292
    print('Legs IG : ', get_information_gain(df,'legs')) #0.5567796494470394
    print(get_selected_attribute(df)) #legs'''
    

if __name__=="__main__":
	test_case()