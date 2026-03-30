# Cloud Function triggered by GCS uploads to the data bucket
# Kicks off the mlops-agents pipeline when new data arrives

resource "google_project_service" "cloudfunctions" {
  project            = var.project_id
  service            = "cloudfunctions.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudbuild" {
  project            = var.project_id
  service            = "cloudbuild.googleapis.com"
  disable_on_destroy = false
}

# GCS bucket for incoming data (trigger source)
resource "google_storage_bucket" "data_upload" {
  name     = "${var.project_id}-mlops-data-${var.environment}"
  location = var.bucket_location
  project  = var.project_id

  uniform_bucket_level_access = true
  force_destroy               = var.environment == "dev" ? true : false

  depends_on = [google_project_service.apis]
}

# Grant the service account access to the data bucket
resource "google_storage_bucket_iam_member" "mlops_agent_data" {
  bucket = google_storage_bucket.data_upload.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.mlops_agent.email}"
}

# Cloud Function source bucket
resource "google_storage_bucket" "function_source" {
  name     = "${var.project_id}-mlops-functions-${var.environment}"
  location = var.bucket_location
  project  = var.project_id

  uniform_bucket_level_access = true
  force_destroy               = true

  depends_on = [google_project_service.apis]
}

# Cloud Function (2nd gen) triggered by GCS upload
resource "google_cloudfunctions2_function" "pipeline_trigger" {
  name     = "mlops-pipeline-trigger-${var.environment}"
  location = var.region
  project  = var.project_id

  build_config {
    runtime     = "python312"
    entry_point = "handle_gcs_upload"

    source {
      storage_source {
        bucket = google_storage_bucket.function_source.name
        object = "cloud-function-source.zip"
      }
    }
  }

  service_config {
    available_memory   = "512M"
    timeout_seconds    = 300
    min_instance_count = 0
    max_instance_count = 5

    service_account_email = google_service_account.mlops_agent.email

    environment_variables = {
      PIPELINE_CONFIG = "gs://${google_storage_bucket.mlops_staging.name}/config/pipeline.yaml"
    }
  }

  event_trigger {
    event_type            = "google.cloud.storage.object.v1.finalized"
    trigger_region        = var.region
    retry_policy          = "RETRY_POLICY_RETRY"

    event_filters {
      attribute = "bucket"
      value     = google_storage_bucket.data_upload.name
    }
  }

  depends_on = [
    google_project_service.cloudfunctions,
    google_project_service.cloudbuild,
  ]
}

# Grant Cloud Functions invoker role
resource "google_project_iam_member" "mlops_agent_functions" {
  project = var.project_id
  role    = "roles/cloudfunctions.invoker"
  member  = "serviceAccount:${google_service_account.mlops_agent.email}"
}

output "data_upload_bucket" {
  value = "gs://${google_storage_bucket.data_upload.name}"
}

output "function_name" {
  value = google_cloudfunctions2_function.pipeline_trigger.name
}
