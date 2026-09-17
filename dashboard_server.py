from flask import Flask, jsonify, request, render_template, send_file
from db import *
import csv, io, os
from datetime import datetime

app=Flask(__name__)
init_db()
ACTIVE_SESSION=None

@app.get('/')
def dashboard(): return render_template('dashboard.html')
@app.get('/analysis')
def analysis(): return render_template('analysis.html')
@app.get('/admin/students')
def admin_students(): return render_template('admin_students.html')
@app.get('/admin/student/<student_id>')
def admin_student(student_id): return render_template('admin_student_detail.html',student_id=student_id)

@app.get('/api/status')
def status():
    return jsonify({r['id']:{'name':r['name'],'score':round(r['score'] or 0,1),'state':r['state'] or 'Waiting','confidence':r['confidence'] or 0,'timestamp':r['timestamp']} for r in latest_students()})

@app.get('/api/timeline')
def api_timeline(): return jsonify(timeline(int(request.args.get('minutes',30)),request.args.get('session_id',type=int)))

@app.post('/api/session/start')
def start_session():
    global ACTIVE_SESSION
    d=request.get_json(silent=True) or {}; sid=d.get('student_id','S1'); name=d.get('name','Student 1'); upsert_student(sid,name); ACTIVE_SESSION=create_session(sid,d.get('mode','study')); return jsonify({'session_id':ACTIVE_SESSION})

@app.post('/api/session/end')
def stop_session():
    global ACTIVE_SESSION
    d=request.get_json(silent=True) or {}; sid=d.get('session_id') or ACTIVE_SESSION
    if sid: end_session(sid); summary=session_summary(sid)
    else: summary=None
    ACTIVE_SESSION=None; return jsonify(summary or {})

@app.post('/api/update')
def update():
    d=request.get_json(silent=True) or {}; sid=d.get('student_id','S1'); name=d.get('name',sid); upsert_student(sid,name)
    session_id=int(d.get('session_id') or ACTIVE_SESSION or create_session(sid))
    score=float(d.get('score',0)); state=d.get('state') or ('Focused' if score>=55 else 'Distracted')
    log_attention(sid,session_id,score,state,d.get('signals')); return jsonify({'success':True,'session_id':session_id})

@app.post('/api/event')
def event():
    d=request.get_json(silent=True) or {}; sid=d.get('student_id','S1'); session_id=int(d.get('session_id') or ACTIVE_SESSION or 0)
    if not session_id: return jsonify({'error':'session_id required'}),400
    log_event(sid,session_id,d.get('event_type','unknown'),d.get('duration',0)); return jsonify({'success':True})

@app.get('/api/summary/<int:session_id>')
def summary(session_id): return jsonify(session_summary(session_id) or {})
@app.get('/api/history/<student_id>')
def history(student_id): return jsonify(student_history(student_id))

@app.get('/api/admin/students')
def students_api():
    conn=get_db(); rows=conn.execute('''SELECT s.id,s.name,COUNT(a.id) records,ROUND(AVG(a.score),1) avg_score FROM students s LEFT JOIN attention_logs a ON a.student_id=s.id GROUP BY s.id ORDER BY s.id''').fetchall(); conn.close(); return jsonify([dict(r) for r in rows])

@app.get('/api/export/csv')
def export_csv():
    output=io.StringIO(); w=csv.writer(output); w.writerow(['Student ID','Name','Timestamp','Score','State','Confidence','Session ID'])
    for r in all_records(): w.writerow([r['student_id'],r['name'],r['timestamp'],r['score'],r['state'],r['confidence'],r['session_id']])
    return send_file(io.BytesIO(output.getvalue().encode()),mimetype='text/csv',as_attachment=True,download_name='FocusSense_Report.csv')

@app.get('/api/export/pdf')
def export_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    buf=io.BytesIO(); pdf=canvas.Canvas(buf,pagesize=A4); y=800
    pdf.setFont('Helvetica-Bold',18); pdf.drawString(40,y,'FocusSense — Attention Report'); y-=28
    pdf.setFont('Helvetica',9)
    for r in all_records():
        line=f"{r['student_id']} | {r['name']} | {r['timestamp']} | {r['score']:.1f}% | {r['state']}"
        pdf.drawString(40,y,line[:115]); y-=13
        if y<45: pdf.showPage(); y=800; pdf.setFont('Helvetica',9)
    pdf.save(); buf.seek(0); return send_file(buf,mimetype='application/pdf',as_attachment=True,download_name='FocusSense_Report.pdf')

@app.get('/api/health')
def health(): return jsonify({'status':'ok','time':datetime.utcnow().isoformat()+'Z'})

if __name__=='__main__': app.run(host='127.0.0.1',port=int(os.getenv('PORT',5000)),debug=False)
