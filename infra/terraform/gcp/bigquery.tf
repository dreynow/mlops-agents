# BigQuery dataset for feature data and audit trail

resource "google_bigquery_dataset" "ml_features" {
  dataset_id    = "ml_features_${var.environment}"
  friendly_name = "ML Features (${var.environment})"
  description   = "Feature tables for ML pipeline training and evaluation"
  location      = var.bq_location
  project       = var.project_id

  # 30-day default table expiration for dev, none for prod
  default_table_expiration_ms = var.environment == "dev" ? 2592000000 : null

  depends_on = [google_project_service.apis]
}

resource "google_bigquery_dataset" "ml_audit" {
  dataset_id    = "ml_audit_${var.environment}"
  friendly_name = "ML Audit Trail (${var.environment})"
  description   = "Decision audit trail from mlops-agents pipeline runs"
  location      = var.bq_location
  project       = var.project_id

  depends_on = [google_project_service.apis]
}

# Grant the service account data editor access
resource "google_bigquery_dataset_iam_member" "mlops_agent_features" {
  dataset_id = google_bigquery_dataset.ml_features.dataset_id
  project    = var.project_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.mlops_agent.email}"
}

resource "google_bigquery_dataset_iam_member" "mlops_agent_audit" {
  dataset_id = google_bigquery_dataset.ml_audit.dataset_id
  project    = var.project_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.mlops_agent.email}"
}

# Also need job user role to run queries
resource "google_project_iam_member" "mlops_agent_bq_job" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.mlops_agent.email}"
}

output "features_dataset" {
  value = google_bigquery_dataset.ml_features.dataset_id
}

output "audit_dataset" {
  value = google_bigquery_dataset.ml_audit.dataset_id
}
