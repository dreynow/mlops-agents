# Cloud Run IAM for model serving
# The actual Cloud Run services are managed by the DeployAgent
# via the CloudRunServing provider. Terraform handles IAM only.

resource "google_project_service" "run" {
  project            = var.project_id
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

# Artifact Registry for model container images
resource "google_project_service" "artifactregistry" {
  project            = var.project_id
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "models" {
  project       = var.project_id
  location      = var.region
  repository_id = "models"
  description   = "Container images for model serving"
  format        = "DOCKER"

  depends_on = [google_project_service.artifactregistry]
}

# Grant the service account Cloud Run developer role
resource "google_project_iam_member" "mlops_agent_run" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.mlops_agent.email}"
}

# Grant Artifact Registry writer (to push model images)
resource "google_artifact_registry_repository_iam_member" "mlops_agent_ar" {
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.models.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.mlops_agent.email}"
}

# Cloud Monitoring reader (for serving metrics)
resource "google_project_iam_member" "mlops_agent_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.viewer"
  member  = "serviceAccount:${google_service_account.mlops_agent.email}"
}

output "artifact_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.models.name}"
}
