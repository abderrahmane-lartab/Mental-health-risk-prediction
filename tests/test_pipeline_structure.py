import ast
from pathlib import Path
import unittest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestPipelineIntegration(unittest.TestCase):
    def test_python_files_compile_cleanly(self):
        python_files = [
            PROJECT_ROOT / "src" / "__init__.py",
            PROJECT_ROOT / "src" / "preprocessing.py",
            PROJECT_ROOT / "src" / "models.py",
            PROJECT_ROOT / "src" / "training.py",
            PROJECT_ROOT / "src" / "optimization.py",
            PROJECT_ROOT / "airflow" / "dags" / "mental_health_pipeline.py",
            PROJECT_ROOT / "app" / "Home.py",
            PROJECT_ROOT / "app" / "components" / "visualizations.py",
            PROJECT_ROOT / "app" / "pages" / "1_Data_Analysis.py",
            PROJECT_ROOT / "app" / "pages" / "2_Model_Performance.py",
            PROJECT_ROOT / "app" / "pages" / "3_Prediction_Demo.py",
        ]
        for file_path in python_files:
            self.assertTrue(file_path.exists(), f"Missing file: {file_path}")
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
            self.assertIsNotNone(tree)

    def test_dag_structure_and_tasks(self):
        dag_path = PROJECT_ROOT / "airflow" / "dags" / "mental_health_pipeline.py"
        source = dag_path.read_text(encoding="utf-8")

        self.assertIn('dag_id="student_mental_health_pipeline"', source)
        expected_task_ids = [
            "data_ingestion",
            "data_preprocessing",
            "model_training",
            "cross_validation",
            "hyperparameter_tuning",
            "model_evaluation",
            "model_saving",
        ]
        for task_id in expected_task_ids:
            self.assertIn(f'task_id="{task_id}"', source)

    def test_docker_compose_valid_yaml(self):
        compose_path = PROJECT_ROOT / "docker-compose.yml"
        self.assertTrue(compose_path.exists())
        with open(compose_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.assertIn("services", config)
        for service in [
            "postgres",
            "airflow-init",
            "airflow-webserver",
            "airflow-scheduler",
        ]:
            self.assertIn(service, config["services"])


if __name__ == "__main__":
    unittest.main()
