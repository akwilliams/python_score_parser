'''
Define the score object
'''

class Score:
  title='na'
  subtitle='na'
  composer='na'
  section=[]
  instrument=[]
  def __init__(self):
    print("Score initialized")

class Section:
  mm_start=-1
  mm_end=-1
  tempo=-1
  key_sig=[-1]
  style_info=[]
  def __init__(self):
    print("Section initialized")

class Instrument:
  name='na'
  family='na'
  part=-1
  measure=[]
  def __init__(self):
    print("Instrument initialized")
    
class Measure:
  note=[]
  dynamics=[]
  performance_technique=[]
  def __init__(self):
    print("Measure initialized")
    
class Note:
  start=[-1,4]
  end=[-1,4]
  articulation='na'
  performance_technique=[]
  def __init__(self):
    print("Note initialized")



















