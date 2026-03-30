# Vertex AI resources for training job management

# Grant the service account Vertex AI user role
resource "google_project_iam_member" "mlops_agent_vertex" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.mlops_agent.email}"
}

# The service account also needs to read/write the staging bucket
# (already granted in gcs.tf via storage.objectAdmin)

# And it needs to use its own SA for training jobs
resource "google_service_account_iam_member" "mlops_agent_self_use" {
  service_account_id = google_service_account.mlops_agent.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.mlops_agent.email}"
}

output "vertex_location" {
  value = var.region
}

output "vertex_service_account" {
  value = google_service_account.mlops_agent.email
}
