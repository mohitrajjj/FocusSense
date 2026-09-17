import sqlite3, os
from datetime import datetime, timezone

DB_PATH = os.getenv('FOCUSSENSE_DB', 'focus.db')

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn

def init_db():
    conn=get_db(); cur=conn.cursor()
    cur.executescript('''
    CREATE TABLE IF NOT EXISTS students(
      id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sessions(
      id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, start_time TEXT NOT NULL,
      end_time TEXT, mode TEXT DEFAULT 'study', FOREIGN KEY(student_id) REFERENCES students(id)
    );
    CREATE TABLE IF NOT EXISTS attention_logs(
      id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT NOT NULL, session_id INTEGER NOT NULL,
      timestamp TEXT NOT NULL, score REAL NOT NULL, state TEXT NOT NULL,
      gaze REAL, head_pose REAL, eye_state REAL, face_presence REAL, confidence REAL,
      FOREIGN KEY(student_id) REFERENCES students(id), FOREIGN KEY(session_id) REFERENCES sessions(id)
    );
    CREATE TABLE IF NOT EXISTS events(
      id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT NOT NULL, session_id INTEGER NOT NULL,
      timestamp TEXT NOT NULL, event_type TEXT NOT NULL, duration REAL DEFAULT 0,
      FOREIGN KEY(student_id) REFERENCES students(id), FOREIGN KEY(session_id) REFERENCES sessions(id)
    );
    CREATE INDEX IF NOT EXISTS idx_attention_session_time ON attention_logs(session_id,timestamp);
    CREATE INDEX IF NOT EXISTS idx_events_session_time ON events(session_id,timestamp);
    '''); conn.commit(); conn.close()

def now(): return datetime.now(timezone.utc).isoformat()

def upsert_student(student_id,name):
    conn=get_db(); conn.execute('INSERT INTO students(id,name,created_at) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name',(student_id,name,now())); conn.commit(); conn.close()

def create_session(student_id=None, mode='study'):
    conn=get_db(); cur=conn.execute('INSERT INTO sessions(student_id,start_time,mode) VALUES(?,?,?)',(student_id,now(),mode)); conn.commit(); sid=cur.lastrowid; conn.close(); return sid

def end_session(session_id):
    conn=get_db(); conn.execute('UPDATE sessions SET end_time=? WHERE id=?',(now(),session_id)); conn.commit(); conn.close()

def log_attention(student_id,session_id,score,state,signals=None):
    signals=signals or {}; conn=get_db(); conn.execute('''INSERT INTO attention_logs(student_id,session_id,timestamp,score,state,gaze,head_pose,eye_state,face_presence,confidence) VALUES(?,?,?,?,?,?,?,?,?,?)''',
      (student_id,session_id,now(),float(score),state,signals.get('gaze'),signals.get('head_pose'),signals.get('eye_state'),signals.get('face_presence'),signals.get('confidence'))); conn.commit(); conn.close()

def log_event(student_id,session_id,event_type,duration=0):
    conn=get_db(); conn.execute('INSERT INTO events(student_id,session_id,timestamp,event_type,duration) VALUES(?,?,?,?,?)',(student_id,session_id,now(),event_type,float(duration))); conn.commit(); conn.close()

def latest_students():
    conn=get_db(); rows=conn.execute('''SELECT s.id,s.name,a.score,a.state,a.timestamp,a.confidence FROM students s LEFT JOIN attention_logs a ON a.id=(SELECT id FROM attention_logs WHERE student_id=s.id ORDER BY id DESC LIMIT 1) ORDER BY s.id''').fetchall(); conn.close(); return rows

def timeline(minutes=30, session_id=None):
    conn=get_db(); where='timestamp >= datetime(\'now\', ?)' # ISO timestamps make SQLite datetime comparison unreliable; use julianday below
    params=[f'-{minutes} minutes']
    sql='''SELECT student_id,timestamp,score,state FROM attention_logs WHERE julianday(timestamp) >= julianday('now', ?)'''
    if session_id: sql+=' AND session_id=?'; params.append(session_id)
    sql+=' ORDER BY timestamp'; rows=conn.execute(sql,params).fetchall(); conn.close()
    out={}
    for r in rows: out.setdefault(r['student_id'],[]).append(dict(r))
    return out

def session_summary(session_id):
    conn=get_db(); s=conn.execute('SELECT * FROM sessions WHERE id=?',(session_id,)).fetchone();
    if not s: conn.close(); return None
    a=conn.execute('SELECT COUNT(*) n, AVG(score) avg_score FROM attention_logs WHERE session_id=?',(session_id,)).fetchone()
    states=conn.execute('SELECT state,COUNT(*) n FROM attention_logs WHERE session_id=? GROUP BY state',(session_id,)).fetchall()
    ev=conn.execute('SELECT event_type,COUNT(*) n,COALESCE(SUM(duration),0) duration FROM events WHERE session_id=? GROUP BY event_type',(session_id,)).fetchall(); conn.close()
    return {'session':dict(s),'samples':a['n'] or 0,'average_score':round(a['avg_score'] or 0,1),'states':{r['state']:r['n'] for r in states},'events':{r['event_type']:{'count':r['n'],'duration':round(r['duration'],1)} for r in ev}}

def student_history(student_id,limit=1000):
    conn=get_db(); rows=conn.execute('SELECT timestamp,score,state,confidence,session_id FROM attention_logs WHERE student_id=? ORDER BY timestamp DESC LIMIT ?',(student_id,limit)).fetchall(); conn.close(); return [dict(r) for r in rows]

def all_records():
    conn=get_db(); rows=conn.execute('''SELECT s.id student_id,s.name,a.timestamp,a.score,a.state,a.confidence,a.session_id FROM attention_logs a JOIN students s ON s.id=a.student_id ORDER BY a.timestamp''').fetchall(); conn.close(); return [dict(r) for r in rows]
