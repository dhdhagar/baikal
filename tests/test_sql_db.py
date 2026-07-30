"""Tests for SQLite cluster materialization."""

import unittest

from src.sql_db import build_cluster_sqlite


class TestBuildClusterSqlite(unittest.TestCase):
    def test_case_insensitive_column_dedup(self):
        """TAT-QA tables may differ only by column name casing (e.g. T1580)."""
        table_metas = {
            "T1580": {
                "columns": [
                    "significant items",
                    "Significant items",
                    "Sales",
                ],
                "rows": [
                    {"significant items": "a", "Significant items": "b", "Sales": "c"},
                ],
            }
        }
        conn, schema = build_cluster_sqlite(table_metas)
        try:
            cur = conn.cursor()
            cur.execute('PRAGMA table_info("T1580")')
            names = [row[1] for row in cur.fetchall()]
            self.assertEqual(len(names), len({n.lower() for n in names}))
            self.assertIn("significant_items", names)
            self.assertIn("Significant_items_2", names)
            cur.execute('SELECT * FROM "T1580"')
            row = cur.fetchone()
            self.assertEqual(row[0], "a")
            self.assertEqual(row[1], "b")
            self.assertEqual(row[2], "c")
            self.assertIn("significant_items", schema)
        finally:
            conn.close()

    def test_exact_duplicate_columns_get_suffix(self):
        table_metas = {
            "T1": {
                "columns": ["Total", "Total", "Other"],
                "rows": [{"Total": "first", "Other": "x"}],
            }
        }
        conn, _ = build_cluster_sqlite(table_metas)
        try:
            cur = conn.cursor()
            cur.execute('PRAGMA table_info("T1")')
            names = [row[1] for row in cur.fetchall()]
            self.assertEqual(names, ["Total", "Total_2", "Other"])
        finally:
            conn.close()

    def test_empty_column_headers_get_unique_names(self):
        table_metas = {
            "T1": {
                "columns": ["", "", "Revenue"],
                "rows": [[1, 2, 3]],
            }
        }
        conn, _ = build_cluster_sqlite(table_metas)
        try:
            cur = conn.cursor()
            cur.execute('PRAGMA table_info("T1")')
            names = [row[1] for row in cur.fetchall()]
            self.assertEqual(names, ["col", "col_2", "Revenue"])
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
