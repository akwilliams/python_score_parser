import abjad as ab
import random

def gen_example(mm_min=2,mm_max=4,sig_dur_max=3):
    mm_count=random.randint(mm_min,mm_max)
    time_sig_count=random.randint(1,mm_count)
    if time_sig_count>1:
        time_sig=[]
        time_sig_index=[]
        for x in range(time_sig_count):
            time_sig.append(ab.TimeSignature((random.randint(1,sig_dur_max),2**(random.randint(0,4)))))
            
            print(time_sig[x].pair)
            if x !=0:
                time_sig_index.append(random.randint(time_sig_index[x-1]+1,mm_count-1-(time_sig_count-1-x)))
            else:
                time_sig_index.append(0)

gen_example()

#[0,1,2,3,4]
#[(prev_time_sig_index+1):-1*len(remaining_time_sigs)]
prev_time_sig_index=0
mm_count=3
remaining_time_sig_count=1
index=random.randint(prev_time_sig_index+1,mm_count-1-remaining_time_sig_count)
print(index)


        
gen_example()


clef=ab.Clef('bass')
clef_spanner=ab.ClefSpanner('treble')
time_signature_0=ab.TimeSignature((2,4))
time_signature_1=ab.TimeSignature((3,8))

components = [ab.Tuplet(ab.Multiplier(2, 3), "c'4 d4 e4"), ab.Note("f8."), ab.Note("g8.")]
staff = ab.Staff(components)
ab.attach(clef,staff)
ab.attach(clef_spanner,staff[-2:])
ab.attach(time_signature_0,staff)
ab.attach(time_signature_1,staff[1])
ab.show(staff)




