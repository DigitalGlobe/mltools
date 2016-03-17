# Contains functions for reading from and writing to the Tomnod database.

import os
import psycopg2


class TomnodCommunicator():


    class DatabaseError(Exception):
        pass


    def __init__(self, credentials):
        '''Args:
               parameters (dict): Dictionary with Tomnod credentials.
        '''                
        self.host = credentials['host']
        self.db = credentials['db']
        self.user = credentials['user']
        self.password = credentials['password']
        if not(''.join([self.host, self.db, self.user, self.password])):
            raise DatabaseError('Can not connect to Tomnod. Credentials missing.')
    

    def _get_connection(self):
        params = 'host={} dbname={} user={} password={}'.format(self.host, 
                                                                self.db, 
                                                                self.user, 
                                                                self.password)
        connection = psycopg2.connect(params)
        connection.autocommit = True
        return connection


    def _fetch(self, query):
        connection = self._get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(query)
            r = cursor.fetchall()
            connection.close()
            return r
        except psycopg2.ProgrammingError, e:
            print 'Programming error in query: {}'.format(e)
            connection.close()
            return


    def _execute(self, query):
        connection = self._get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(query)
        except psycopg2.ProgrammingError, e:
            print 'Programming error in query: {}'.format(e)
        connection.close()
        return       


    def get_high_confidence_features(self,
                                     campaign_schema, 
                                     image_id,
                                     class_name,
                                     max_number = 1000, 
                                     min_score = 0.95, 
                                     min_votes = 0,
                                     max_area = 1e06,
                                    ):
        '''Read high-confidence data from a Tomnod classification campaign for 
           given image_id and given class. Features are read in decreasing score 
           order. The purpose of this function is to create training/test 
           data for a machine.
           
           Args:
               campaign_schema (str): Campaign campaign_schema.
               image_id (str): Image id.
               class_name (str): Feature class (type in Tomnod jargon) name.
               max_number (int): Maximum number of features to be read.
               min_score (float): Only features with score>=min_score will be read.
               min_votes (int): Only features with votes>=min_votes will be read.
               max_area (float): Only import features with (area in m2) <= max_area.

           Returns:
               A list of tuples (feature_coordinates_in_hex, feature_id, image_id, class_name).    
        '''

        query = '''SELECT feature.feature, feature.id, overlay.catalogid, tag_type.name
                   FROM {}.feature, tag_type, overlay
                   WHERE feature.type_id = tag_type.id
                   AND feature.overlay_id = overlay.id     
                   AND overlay.catalogid = '{}'
                   AND tag_type.name = '{}'
                   AND feature.score >= {}
                   AND feature.num_votes_total >= {}
                   AND ST_Area(feature.feature) <= {}
                   ORDER BY feature.score DESC LIMIT {}'''.format(campaign_schema, 
                                                                  image_id, 
                                                                  class_name, 
                                                                  min_score,
                                                                  min_votes,
                                                                  max_area,
                                                                  max_number)
        
        return self._fetch(query)

        
    def get_low_confidence_features(self,
                                    campaign_schema, 
                                    image_id,
                                    max_number = 1000,
                                    max_score = 1.0,
                                    max_votes = 0,
                                    max_area = 1e06,
                                   ):

        '''Read low-confidence data from a Tomnod classification campaign for a 
           given image_id. Features are read from the DB in increasing score order, 
           nulls first. (A feature with null score has not had its score computed 
           by crowdrank.) The purpose of this function is to create target data 
           for a machine.
       
           Args:
               campaign_schema (str): Campaign campaign_schema.
               image_id (str): Image id.
               max_number (int): Maximum number of features to be read.
               max_score (float): Only features with score<=max_score will be read.
               max_votes (int): Only features with votes<=max_votes will be read.
               max_area (float): Only import features with (area in m2) <= max_area.

           Returns:
               A list of tuples (feature_coordinates_in_hex, feature_id, image_id).    
        '''

        query = """SELECT feature.feature, feature.id, overlay.catalogid
                   FROM {}.feature, overlay
                   WHERE feature.overlay_id = overlay.id        
                   AND overlay.catalogid = '{}'
                   AND (feature.score <= {} OR feature.score IS NULL)
                   AND feature.num_votes_total <= {}
                   AND ST_Area(feature.feature) <= {}
                   ORDER BY feature.score ASC NULLS FIRST
                   LIMIT {}""".format(campaign_schema, 
                                      image_id,  
                                      max_score,
                                      max_votes,
                                      max_area, 
                                      max_number)          

        return self._fetch(query)


    def execute(self, query, data, batch_size = 1000):
        '''Execute query for each entry in data.
           Args:
               query (str): SQL query with {} for arguments.
               data (list): Data to format sql query. Each entry is a tuple.
               batch_size (int): Execute in batches of batch_size.  
        '''
        total_query, no_entries = '', len(data)
        for i, entry in enumerate(data):
            total_query += query.format(*entry)
            if (i%(batch_size-1)  == 0) or (i == no_entries-1):
                self._execute(total_query)
                total_query = ''
