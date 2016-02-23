'''
This module contains a collection of functions for 
connecting with and running queries on the Tomnod DB.
'''

import psycopg2
import psycopg2.extras
import os

# DB parameters:
host = os.getenv( 'DB_HOST', None )
db = os.getenv( 'DB_NAME', None )
user = os.getenv( 'DB_USER', None )
password = os.getenv( 'DB_PASSWORD', None )
port = os.getenv( 'DB_PORT', None )

class DatabaseError(Exception):
    pass

def getConn(host=host,db=db,port=port,user=user,password=password):
    if user == '' or password == '':
        raise DatabaseError('Database username or password not set.')

    conn_string = "host=%s dbname=%s user=%s password=%s" % (host, db, user, password)
    #print "Connecting to database\n	->%s" % (conn_string)
    conn = psycopg2.connect(conn_string)
    conn.autocommit = True
    return conn

def getCursor( conn, fetch_array = True ):
    if fetch_array:
    	return conn.cursor()
    else:
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# Use db_query for queries that return nothing
def db_query( sql ):
    conn = getConn()
    cursor = getCursor(conn, False)
    try:
        cursor.execute( sql )
    except psycopg2.ProgrammingError, e:
        print "Programming error in query:\n%s" % e
	conn.close()
	return

def db_fetch( sql, fetch_array = True ):
    conn = getConn()
    cursor = getCursor(conn, fetch_array)
    try:
        cursor.execute( sql )
        r = cursor.fetchall()
        conn.close()
        return r
    except psycopg2.ProgrammingError, e:
        print "Programming error in query:\n%s" % e
        conn.close()
        return 

def db_fetch_dict( sql ):
    return db_fetch( sql, False )

def db_fetch_array( sql ):
    return db_fetch( sql, True )

if __name__ == "__main__":
    sql = "SELECT id, description from overlay order by id desc limit 10;"
    results = db_fetch_array( sql )
    for (id, description) in results:
        print id, description

    results = db_fetch_dict( sql )
    print results
