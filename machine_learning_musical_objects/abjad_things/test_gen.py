import abjad as ab
import random
import math

options_clef=['treble','treble^8','treble^15','alto','tenor','baritone','varbariton','mezzosprano','soprano','bass','percussion','tab','blackmensural-c1','blackmensural-c2','blackmensural-c3','blackmensural-c4','blackmensural-c5','mensural-c1','mensural-c2','mensural-c3','mensural-c4','mensural-c5','mensural-f','mensural-g','petrucci-c1','petrucci-c2','petrucci-c3','petrucci-c4','petrucci-c5','petrucci-f','petrucci-g','neomensural-c1','neomensural-c2','neomensural-c3','neomensural-c4','neomensural-c5']
options_notehead=['default','altdefault','baroque','neomensural','mensural','petrucci','harmonic','harmonic-black','harmonic-mixed','diamond','cross','xcircle','triangle','slash']

def gen_example(mm_min=2,mm_max=4,sig_dur_max=5,min_notes_in_mm=1,max_notes_in_mm=8,clef_max=3,sys_min=1,sys_max=6,pitchset=[0,2,4,5,7,9,11]):
    #Calc the number of measures
    mm_count=random.randint(mm_min,mm_max)
    #Calc the number of systems
    sys_count=random.randint(sys_min,sys_max)
    #Calc the number of time signatures
    time_sig_count=random.randint(1,mm_count)
    time_sig=[]
    time_sig_index=[]
    #Calc the time signatures and their index within the measures
    for x in range(time_sig_count):
        time_sig.append(ab.TimeSignature((random.randint(1,sig_dur_max),2**(random.randint(0,4)))))
        if x !=0:
            time_sig_index.append(random.randint(time_sig_index[x-1]+1,mm_count-1-(time_sig_count-1-x)))
        else:
            time_sig_index.append(0)
    #Calc the number of clefs
    clef_count=random.randint(1,clef_max)
    clef=[]
    for x in range(clef_count):
        clef_src=options_clef[random.randint(0,(len(options_clef)-1))]
        if x != 0:
            clef.append(ab.ClefSpanner(clef_src))
        else:
            clef.append(ab.Clef(clef_src))
    #For each system generate musical content
    systems=[]
    for i in range(sys_count):
        #Initialize each system
        systems.append([])
        #For each time sig change
        for x in range(len(time_sig)):
            #if the next time sig is not the following measure, calculate how many measures inbetween the current and following
            if x+1==len(time_sig_index):
                #If this is the final time sig change calc remaining measures in example
                mm_iter_count=mm_count-time_sig_index[x]
            elif time_sig_index[(x+1)]!=(x+1):
                #If the following time sig has an index!= to the following measure, clac the number of measures with this time sig
                mm_iter_count=time_sig_index[(x+1)]-time_sig_index[x]
            else:
                #If the following measure has a time sig change, iter count == 1 
                mm_iter_count=1
            #For each of the measures to be generated
            for y in range(mm_iter_count):
                systems[i].append(gen_measure(time_sig=time_sig[x].pair,pitchset=pitchset))
    return systems
            
def gen_measure(time_sig=(4,4),note_count_min=1,note_count_max=12,pitchset=range(11),min_dur=(1/16),subdivision=[1,2,4,8,16,32]):
    content=[]
    note_count=random.randint(note_count_min,note_count_max)
    #print(note_count)
    #generate a subdivision and a number of those subdivisions to generate note durations
    #must have a max and min duration
    max_dur = (time_sig[0]/time_sig[1])-(note_count)*min_dur
    dur_used=0
    for xx in range(note_count):
        max_dur=(time_sig[0]/time_sig[1])-dur_used-(note_count-(xx+1))*min_dur
        if xx!=note_count-1:
            dur,dur_flt=gen_note_dur(min_dur,max_dur,list(subdivision))
        else:
            #if it is the final duration the min and max dur must be the remaining duration
            dur,dur_flt=gen_note_dur(min_dur,max_dur,list(subdivision))
        dur_used=dur_used+dur_flt
        note_lvl=random.randint(0,11)
        #If generated pitch is not in pitchset add a semitone until it is
        while note_lvl not in pitchset:
            note_lvl=note_lvl+1
            if note_lvl>11:
                note_lvl=note_lvl-12
        #Add octave offsets
        note_lvl=note_lvl+(random.randint(-1,1)*12)
        for duration in dur:
            content.append(ab.Note(note_lvl,duration))
    #This is where a logical algorithm should be to check rhythm things
    from random import shuffle
    shuffle(content)
    return content

def gen_note_dur(dur_min,dur_max,subdivision):
    hold=[]
    print(dur_max)
    for val in subdivision:
        if 1/val>dur_max:
            hold.append(val)
    subdivision = [yy for yy in subdivision if yy not in hold]
    print(subdivision)
    denominator=subdivision[random.randint(0,len(subdivision)-1)]
    numerator_max=-1
    for yy in range(math.ceil(dur_max)*denominator+2):
        if yy/denominator>dur_max and numerator_max==-1:
            numerator_max=yy-1
    numerator=random.randint(1,numerator_max)
    dur=[]
    index=0
    if numerator in [5,7,9,10,11,13,15,17,19,21,23,29,31]:
        while numerator>4:
            index=index+1
            numerator=numerator-4
        dur.append(ab.Duration(index*4,denominator))
    dur.append(ab.Duration(numerator,denominator))
    return dur,((numerator+(4*index))/denominator)

time_sig_=(9,8)
staff=ab.Staff(gen_measure(time_sig=time_sig_,note_count_min=3,note_count_max=14,pitchset=[1,2,4,6,7,9,11],min_dur=(1/16)))
time_sig_o=ab.TimeSignature(time_sig_)
ab.attach(time_sig_o,staff)
ab.show(staff)

systems = gen_example(sys_max=2,sys_min=2)
print(systems)
#staff=ab.Staff(systems[0][0])
staff_0=ab.Staff(systems[0][0])
staff_1=ab.Staff(systems[1][0])

score=ab.Score([staff_0,staff_1])
ab.show(score)
'''
#special noteheads are created by doing the following
#ab.override(staff[notehead_index]).note_head.style=notehead_options[0]

        
gen_example()


clef=ab.Clef('tenor')
clef_spanner=ab.ClefSpanner('treble')
time_signature_0=ab.TimeSignature((2,4))
time_signature_1=ab.TimeSignature((3,8))

components = [ab.Tuplet(ab.Multiplier(2, 3), "c'4 d4 e4"), ab.Note("f8."), ab.Note("g8.")]
staff = ab.Staff(components)
for x in range(len(staff)):
    ab.override(staff[x]).note_head.style=options_notehead[random.randint(0,len(options_notehead))]
ab.attach(clef,staff)
ab.attach(clef_spanner,staff[-2:])
ab.attach(time_signature_0,staff)
ab.attach(time_signature_1,staff[1])
staff_1=ab.Staff("c'8 d'8 e'8 f'8")
score=ab.Score([staff,staff_1])
ab.show(score)

'''


