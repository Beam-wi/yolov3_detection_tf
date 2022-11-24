# coding=utf-8

import os
import pymysql
import configparser
from DBUtils.PersistentDB import PersistentDB
# from config.ConfigReadUtil import read


class DbUtil:

    def __init__(self):
        self.host = read("mysql", "host")
        self.user = read("mysql", "user")
        self.password = read("mysql", "password")
        self.port = int(read("mysql", "port"))
        self.db = read("mysql", "db")
        self.db_pool = self.makePool()

    # 获取数据库连接池里的连接
    def _getconnection(self):
        # 从数据库连接池是取出一个数据库连接
        conn = self.db_pool.connection()
        return conn

    # 插入
    def insert(self, sql, params):
        conn = self._getconnection()
        id = 0
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            id = cursor.lastrowid
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

        return id

    # 查询
    def queryAll(self, sql, params):
        conn = self._getconnection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            result = cursor.fetchall()
            return result
        except Exception as e:
            raise e
        finally:
            conn.close()

    def queryOne(self, sql, params=None):
        dataOne = None
        conn = self._getconnection()
        try:
            cur = conn.cursor()
            count = cur.execute(sql, params)
            if count != 0:
                dataOne = cur.fetchone()
        except Exception as e:
            raise e
        finally:
            conn.close()

        return dataOne

    # 创建数据库连接池
    def makePool(self):
        config = {
            'host': self.host,
            'port': self.port,
            'database': self.db,
            'user': self.user,
            'password': self.password,
            'charset': 'utf8'
        }
        return PersistentDB(pymysql, **config)

    # 更新
    def update(self, sql, params):
        conn = self._getconnection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    # 批量插入
    def batch_insert(self, sql, params):
        conn = self._getconnection()
        try:
            cursor = conn.cursor()
            cursor.executemany(sql, params)
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    # 批量更新
    def batch_update(self, sql, params):
        conn = self._getconnection()
        try:
            cursor = conn.cursor()
            cursor.executemany(sql, params)
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

def read(key, key1):
    proDir = os.path.split(os.path.realpath(__file__))[0]
    # proDir = os.path.dirname(os.path.realpath(__file__))  与上面一行代码作用一样
    configPath = os.path.join(proDir, "../config", "config.cfg")
    path = os.path.abspath(configPath)
    conf = configparser.ConfigParser()
    # 下面3种路径方式都可以
    conf.read(path)
    value = conf.get(key, key1)

    return value

# 单例对象
# dbService = DbUtil()