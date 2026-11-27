import io, os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pandas as pd
from app import create_app

SAMPLE = pd.read_csv(os.path.join(ROOT, "data", "sensor_data.csv"))


def upload(client, df, name="d.csv"):
    return client.post("/api/score", data={"file": (io.BytesIO(df.to_csv(index=False).encode()), name)}, content_type="multipart/form-data")


class TestWeb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = create_app().test_client()

    def test_pages(self):
        self.assertEqual(self.c.get("/").status_code, 200)
        self.assertEqual(self.c.get("/api/health").get_json()["status"], "ok")
        for path in ("/static/app.js", "/static/style.css"):
            self.assertEqual(self.c.get(path).status_code, 200)

    def test_model_and_units(self):
        m = self.c.get("/api/model").get_json()
        self.assertEqual(len(m["config"]["features"]), 9); self.assertGreater(m["config"]["threshold"], 0)
        self.assertIn("detection_rate", m["autoencoder"])
        units = self.c.get("/api/units").get_json()
        self.assertEqual(len(units), 60); self.assertTrue(units[0]["is_test"])
        self.assertEqual(sum(u["is_test"] for u in units), 15)

    def test_unit_detail(self):
        d = self.c.get("/api/unit/3").get_json()
        n = len(d["cycles"])
        for key in ("status", "health", "rul_estimate", "true_rul"):
            self.assertEqual(len(d[key]), n)
        self.assertEqual(len(d["sensors"]["vibration"]), n); self.assertEqual(d["true_rul"][-1], 0)
        self.assertEqual(d["status"][-1], "critical"); self.assertIn("fault", d["truth"])
        self.assertEqual(self.c.get("/api/unit/9999").status_code, 404)

    def test_score_upload(self):
        r = upload(self.c, SAMPLE[SAMPLE.unit.isin([3, 5])])
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertEqual([u["unit"] for u in d["units"]], [3, 5]); self.assertEqual(d["units"][0]["final_status"], "critical")
        self.assertTrue(d["units"][0]["top_sensors"]); self.assertIn("health_index", d["csv"])
        self.assertEqual(len(pd.read_csv(io.StringIO(d["csv"]))), (SAMPLE.unit.isin([3, 5])).sum())

    def test_upload_without_unit_and_cycle_columns(self):
        one = SAMPLE[SAMPLE.unit == 7].drop(columns=["unit", "cycle"])
        r = upload(self.c, one)
        self.assertEqual(r.status_code, 200); self.assertEqual(r.get_json()["units"][0]["cycles"], len(one))

    def test_upload_errors(self):
        self.assertEqual(self.c.post("/api/score", data={}).status_code, 400)
        r = upload(self.c, SAMPLE.drop(columns=["vibration"]).head(30))
        self.assertEqual(r.status_code, 400); self.assertIn("vibration", r.get_json()["error"])
        r = self.c.post("/api/score", data={"file": (io.BytesIO(b""), "e.csv")}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        r = self.c.post("/api/score", data={"file": (io.BytesIO(bytes([255, 254, 0, 1, 2])), "x.csv")}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
