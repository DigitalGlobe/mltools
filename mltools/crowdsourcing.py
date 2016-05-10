# Contains functions for reading from and writing to the Tomnod database.

import psycopg2


class TomnodCommunicator():


    class DatabaseError(Exception):
        pass


    def __init__(self, credentials):
        """Args:
               parameters (dict): Dictionary with Tomnod credentials.
        """                
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


    def batch_execute(self, query, data, batch_size=1000):
        """Execute query for each entry in data in batches.

           Args:
               query (str): SQL query with {} for arguments.
               data (list): Data to format sql query. Each entry is a tuple.
               batch_size (int): Execute in batches of batch_size.  
        """
        
        total_query, no_entries = '', len(data)
        for i, entry in enumerate(data):
            total_query += query.format(*entry)
            if (i%(batch_size-1)  == 0) or (i == no_entries-1):
                self._execute(total_query)
                total_query = ''


    def get_tags(self,
                 class_name,
                 campaign_schema,
                 most_confident_first=True,
                 score_range=[0.0,1.0],
                 agree_range=[1,10000],
                 image_id=None,
                 max_number=10000):
        """Get tag info from Tomnod extraction campaign for a given class (tag type).
           
           Args:
               class_name (str): Tag class name (tag type in Tomnod jargon).
               campaign_schema (str): Campaign campaign_schema.
               most_confident_first (bool): If True (False), order by decreasing 
                                            (increasing) score, agreement.
               image_id (str): Catalog id. If None, read from all campaign images.
               score_range (list): Min score and max score.
               agree_range (list): Min and max numbers of agreeing tags in cluster.
               max_number (int): Maximum number of tags to be read.

           Returns:
               A list of tuples (coords_in_hex, tag_id, image_id, class_name).    
        """
        
        if most_confident_first:
            which_order = 'DESC'
        else:
            which_order = 'ASC'    

        if image_id is None:
            extra_query = ''
        else:
            extra_query = "AND overlay.catalogid = '{}' ".format(image_id)

        query = ("""SELECT co.point, co.tag_id, overlay.catalogid, tag_type.name
                    FROM {}.crowdrank_output co, tag_type, overlay
                    WHERE co.type_id = tag_type.id
                    AND co.overlay_id = overlay.id """.format(campaign_schema) +     
                    extra_query +
                 """AND tag_type.name = '{}'
                    AND co.cr_score BETWEEN {} AND {}
                    AND co.agreement BETWEEN {} AND {}
                    AND co.job_id = (SELECT MAX(cj.id) 
                                     FROM crowdrank_jobs cj, campaign cn 
                                     WHERE cj.campaign_id = cn.id
                                     AND cn.schema = '{}')
                    ORDER BY co.cr_score {}, co.agreement {} 
                    LIMIT {}""".format(class_name, 
                                       score_range[0],
                                       score_range[1],
                                       agree_range[0],
                                       agree_range[1],
                                       campaign_schema,
                                       which_order,
                                       which_order,
                                       max_number))
        return self._fetch(query)

   
    def get_classified(self,
                       class_name,
                       campaign_schema,
                       most_confident_first=True,
                       score_range=[0.0,1.0],
                       vote_range=[1,10000],
                       image_id=None,
                       max_number=10000,
                       max_area=1e06):
        """Get classified feature info from Tomnod classification campaign.
        
           Args:
               class_name (str): Class (type in Tomnod jargon) name.
               campaign_schema (str): Campaign campaign_schema.
               most_confident_first (bool): If True (False), order by decreasing 
                                            (increasing) score, votes.
               score_range (list): Min score and max score.
               vote_range (list): Min votes and max votes.
               image_id (str): Catalog id. If None, read from all campaign images.
               max_number (int): Maximum number of features to be read.
               max_area (float): Only features with (area in m2) <= max_area.

               Returns:
                   A list of tuples (coords_in_hex, feature_id, image_id, class_name).                                  
        """    

        if most_confident_first:
            which_order = 'DESC'
        else:
            which_order = 'ASC'    

        if image_id is None:
            extra_query = ''
        else:
            extra_query = "AND overlay.catalogid = '{}' ".format(image_id)

        query = ("""SELECT f.feature, f.id, overlay.catalogid, tag_type.name
                    FROM {}.feature f, tag_type, overlay
                    WHERE f.overlay_id = overlay.id """.format(campaign_schema) +
                 extra_query +
                 """AND f.type_id = tag_type.id
                    AND tag_type.name = '{}'
                    AND ST_Area(f.feature) <= {}
                    AND score BETWEEN {} AND {}
                    AND num_votes_total BETWEEN {} AND {}
                    ORDER BY score {}, num_votes_total {}
                    LIMIT {}""".format(class_name,  
                                       max_area, 
                                       score_range[0],
                                       score_range[1],
                                       vote_range[0],
                                       vote_range[1],
                                       which_order,
                                       which_order,
                                       max_number))

        return self._fetch(query)


    def get_unclassified(self,
                         campaign_schema,
                         image_id=None,
                         max_number=10000,
                         max_area=1e06):
        """Get unclassified feature info from Tomnod classification campaign.
        
           Args:
               campaign_schema (str): Campaign campaign_schema.
               image_id (str): Catalog id. If None, read from all campaign images.
               max_number (int): Maximum number of features to be read.
               max_area (float): Only features with (area in m2) <= max_area.

               Returns:
                   A list of tuples (coords_in_hex, feature_id, image_id).                                  
        """    

        if image_id is None:
            extra_query = ''
        else:
            extra_query = "AND overlay.catalogid = '{}' ".format(image_id)

        query = ("""SELECT f.feature, f.id, overlay.catalogid
                   FROM {}.feature f, overlay
                   WHERE f.overlay_id = overlay.id """.format(campaign_schema) + 
                 extra_query +
                 """AND type_id IS NULL
                    AND ST_Area(f.feature) <= {}
                    LIMIT {}""".format(max_area, max_number))

        return self._fetch(query)                   