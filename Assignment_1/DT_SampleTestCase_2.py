from Assignment1 import *   

def test_case():
    print("Testcase 1 :")
    df = pd.read_csv('tests/Test1.csv')
    print('Dataset entropy : ',get_entropy_of_dataset(df)==0.9709505944546686)  #0.9709505944546686
    print('Sky avg info : ',get_entropy_of_attribute(df, 'Sky')==0.9509775004326937) #0.9509775004326937
    print('Sky IG : ', get_information_gain(df, 'Sky')==0.01997309402197489) #0.01997309402197489
    print('Airtemp avg info : ',get_entropy_of_attribute(df, 'Airtemp')==0.6490224995673063) #0.6490224995673063
    print('Airtemp IG : ', get_information_gain(df, 'Airtemp')==0.3219280948873623) #0.3219280948873623
    print('Humidity avg info : ',get_entropy_of_attribute(df, 'Humidity')==0.9509775004326937) #0.9509775004326937
    print('Humidity IG : ', get_information_gain(df, 'Humidity')==0.01997309402197489) #0.01997309402197489
    print('Water avg info : ',get_entropy_of_attribute(df, 'Water')==0.8) #0.8
    print('Water IG : ', get_information_gain(df, 'Water')==0.17095059445466854) #0.17095059445466854
    print('Forecast avg info : ',get_entropy_of_attribute(df, 'Forecast')==0.9509775004326937) #0.9509775004326937
    print('Forecast IG : ', get_information_gain(df, 'Forecast')==0.01997309402197489) #0.01997309402197489
    print(get_selected_attribute(df)) #Airtemp
    print()
    
    print("Testcase 2 :")
    df = pd.read_csv('tests/Test2.csv')
    print('Dataset entropy : ', get_entropy_of_dataset(df)==0.9402859586706311)  #0.9402859586706311
    print('Age avg info : ', get_entropy_of_attribute(df, 'Age')==0.6324823551623816) #0.6324823551623816
    print('Age IG : ', get_information_gain(df, 'Age')==0.30780360350824953) #0.30780360350824953
    print('Income avg info : ', get_entropy_of_attribute(df, 'Income')==0.9110633930116763) #0.9110633930116763
    print('Income IG : ', get_information_gain(df, 'Income')==0.02922256565895487) #0.02922256565895487
    print('Student avg info : ', get_entropy_of_attribute(df, 'Student')==0.7884504573082896) #0.7884504573082896
    print('Student IG : ', get_information_gain(df, 'Student')==0.15183550136234159) #0.15183550136234159
    print('Credit_rating avg info : ', get_entropy_of_attribute(df, 'Credit_rating')==0.8921589282623617) #0.8921589282623617
    print('Credit_rating IG : ', get_information_gain(df, 'Credit_rating')==0.04812703040826949) #0.04812703040826949
    print(get_selected_attribute(df)) #Age
    print()
    
    
    print("Testcase 3 :")
    df = pd.read_csv('tests/Test3.csv')
    print('Dataset entropy : ', get_entropy_of_dataset(df)==0.9852281360342515) #0.9852281360342515
    print('Salary avg info : ', get_entropy_of_attribute(df,'salary')==0.5156629249195446) #0.5156629249195446
    print('Salary IG : ', get_information_gain(df,'salary')==0.46956521111470695) #0.46956521111470695
    print('Location avg info : ', get_entropy_of_attribute(df,'location')==0.2857142857142857) #0.2857142857142857
    print('Location IG : ', get_information_gain(df,'location')==0.6995138503199658) #0.6995138503199658
    print(get_selected_attribute(df)) #location
    print()
    

    print("Testcase 4 :")
    df = pd.read_csv('tests/Test4.csv')
    print('Dataset entropy : ', get_entropy_of_dataset(df)==0.9709505944546686) #0.9709505944546686
    print('Toothed avg info : ', get_entropy_of_attribute(df,'toothed')==0.963547202339972) #0.963547202339972
    print('Toothed IG : ', get_information_gain(df,'toothed')==0.007403392114696539) #0.007403392114696539
    print('Breathes avg info : ', get_entropy_of_attribute(df,'breathes')==0.8264662506490407) #0.8264662506490407
    print('Breathes IG : ', get_information_gain(df,'breathes')==0.1444843438056279) #0.1444843438056279
    print('Legs avg info : ', get_entropy_of_attribute(df,'legs')==0.4141709450076292) #0.4141709450076292
    print('Legs IG : ', get_information_gain(df,'legs')==0.5567796494470394) #0.5567796494470394
    print(get_selected_attribute(df)) #legs
    print()

    print('Testcase 5 : ')
    df = pd.read_csv('tests/Test5.csv')
    print('Dataset entropy : ', get_entropy_of_dataset(df)==1.7295739585136223) #1.7295739585136223
    print('Category avg info : ', get_entropy_of_attribute(df,'category')==0.9182958340544896) #0.9182958340544896
    print('Category IG : ', get_information_gain(df, 'category')==0.8112781244591327) #0.8112781244591327
    print()
    
    print('Testcase 6 : ')
    df = pd.read_csv('tests/Test6.csv')
    print('Dataset entropy : ', get_entropy_of_dataset(df)==0.9709505944546686) #0.9709505944546686
    print('color avg info : ', get_entropy_of_attribute(df,'color')==0.9709505944546686) #0.9709505944546686
    print('color IG : ', get_information_gain(df,'color')==0.0) #0.0
    print('size avg info : ', get_entropy_of_attribute(df,'size')==0.9709505944546686)#0.9709505944546686
    print('size IG : ', get_information_gain(df,'size')==0.0)#0.0
    print('act avg info : ', get_entropy_of_attribute(df,'act')==0.5509775004326937)#0.5509775004326937
    print('act IG : ', get_information_gain(df,'act')==0.4199730940219749)#0.4199730940219749
    print('age avg info : ', get_entropy_of_attribute(df,'age')==0.5509775004326937) #0.5509775004326937
    print('age IG : ', get_information_gain(df,'age')==0.4199730940219749)#0.4199730940219749
    print(get_selected_attribute(df)) #age/act
    print()

    print('Testcase 7 : ')
    df = pd.read_csv('tests/Test7.csv')
    print('Dataset entropy : ', get_entropy_of_dataset(df)==1.438862875041894) #1.438862875041894
    print('caprice avg info : ', get_entropy_of_attribute(df,'caprice')==1.2921917219486116) #1.2921917219486116
    print('caprice IG : ', get_information_gain(df,'caprice')==0.14667115309328227) #0.14667115309328227
    print('topic avg info : ', get_entropy_of_attribute(df,'topic')==1.0883213898497568)#1.0883213898497568
    print('topic IG : ', get_information_gain(df,'topic')==0.3505414851921371)#0.3505414851921371
    print('lmt avg info : ', get_entropy_of_attribute(df,'lmt')==1.4134166106032882)#1.4134166106032882
    print('lmt IG : ', get_information_gain(df,'lmt')==0.02544626443860576)#0.02544626443860576
    print('lpss avg info : ', get_entropy_of_attribute(df,'lpss')==1.3920622169341164) #1.3920622169341164
    print('lpss IG : ', get_information_gain(df,'lpss')==0.04680065810777756)#0.04680065810777756
    print('pb avg info : ', get_entropy_of_attribute(df,'pb')==1.3591269643656734) #1.3591269643656734
    print('pb IG : ', get_information_gain(df,'pb')==0.07973591067622054)#0.07973591067622054
    print(get_selected_attribute(df)) #topic

    print('Testcase 9 : ')
    df = pd.read_csv('tests/Test9.csv')
    print('Dataset entropy : ',get_entropy_of_dataset(df)==0.9182958340544896) #0.9182958340544896
    print('Category avg info : ', get_entropy_of_attribute(df,'category')==0.9182958340544896) #0.9182958340544896
    print('Category IG : ', get_information_gain(df, 'category')==0.0)
    print(get_selected_attribute(df)) #({'category': 0.0}, 'category')

    

if __name__=="__main__":
	test_case()
