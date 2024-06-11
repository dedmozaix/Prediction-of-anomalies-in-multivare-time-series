import psycopg2

class DBConnector:
    def __init__(self, dataset_name, db_connector_type, dataset_type, pred_horizon, db_limit=200):
        super(DBConnector, self).__init__()

        self.connection = None
        self.host = db_connector_type.value["host"]
        self.user = db_connector_type.value["user"]
        self.password = db_connector_type.value["password"]
        self.port = db_connector_type.value["port"]

        self.idx_column = "idx"
        self.db_name = dataset_name
        self.table_name = str(dataset_name).lower() + "_" + str(dataset_type)

        self.db_limit = db_limit
        self.pred_horizon = pred_horizon

        self.create_connection()

    def create_connection(self):
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                dbname=self.db_name
            )
        except psycopg2.OperationalError as e:
            print("ERROR connecting to DB", e)



    def close_connection(self):
        self.connection.close()

    def get_db(self):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * from {} order by {} limit {};".format(self.table_name, self.idx_column, self.db_limit)
                )
                return cursor.fetchall()
        except Exception as ex:
            print("Got DB extraction error", ex)

    def get_preds(self, n_time_steps):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM {} order by {} OFFSET {} ROWS FETCH NEXT {} "
                    "ROWS ONLY;".format(self.table_name, self.idx_column, self.db_limit, n_time_steps)
                )
                # print(cursor.fetchall())
                return cursor.fetchall()
        except Exception as ex:
            print("Got DB preds extraction error", ex)

    def get_column_count(self):

        try:
            self.connection = psycopg2.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                dbname=self.db_name
            )
        except psycopg2.OperationalError as e:
            print("ERROR connecting to DB", e)

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT count(*) FROM information_schema.columns "
                    "WHERE table_name = '{}';".format(self.table_name)
                )
                return int(cursor.fetchone()[0])
        except Exception as ex:
            print("Got DB extraction error", ex)


class DBWriter:
    def __init__(self, database_name, db_connector_type):
        self.db_name = database_name
        self.host = db_connector_type.value["host"]
        self.user = db_connector_type.value["user"]
        self.password = db_connector_type.value["password"]
        self.port = db_connector_type.value["port"]

        self.table_name = db_connector_type.value["table_name"]
        self.create_connection()

    def create_connection(self):
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                dbname=self.db_name
            )
        except Exception as ex:
            print("ERROR connecting to DB", ex)

    def create_new_table(self):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "CREATE TABLE if not exists {} (dataset_type VARCHAR(255) NOT NULL,"
                    "model_name VARCHAR(255) NOT NULL,"
                    "anomaly_detector VARCHAR(255) NOT NULL, "
                    "pred_window Integer NOT NULL,"
                    "num_parameters Integer NOT NULL,"
                    "num_epochs Integer NOT NULL,"
                    "mae_accuracy Float NOT NULL,"
                    "smape_accuracy Float NOT NULL,"
                    "mape_accuracy Float NOT NULL,"
                    "training_time Float NOT NULL,"
                    "prediction_time Float NOT NULL,"
                    "detection_time Float NOT NULL,"
                    "cpu_count Integer NOT NULL,"
                    "cpu_usage Float NOT NULL,"
                    "disk_usage Float NOT NULL,"
                    "used_memory VARCHAR(255) NOT NULL);".format(self.table_name)
                )

        except Exception as ex:
            print("Got table creation error", ex)
            pass

    def insert_into_table(self, args):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO {} (dataset_type, model_name, anomaly_detector, pred_window, num_epochs, num_parameters, mae_accuracy,"
                    "smape_accuracy, mape_accuracy, training_time, prediction_time, detection_time, cpu_count, cpu_usage, disk_usage, used_memory) "
                    "VALUES (%s, %s, %s, %s, %s, %s , %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)".format(self.table_name), args
                )
                self.connection.commit()
        except Exception as ex:
            print("Got DB insertion error", ex)
