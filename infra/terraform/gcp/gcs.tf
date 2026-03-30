# GCS bucket for staging artifacts (models, data, reports)
# Other resources depend on this for artifact storage.

resource "google_storage_bucket" "mlops_staging" {
  name     = "${var.project_id}-mlops-staging-${var.environment}"
  location = var.bucket_location
  project  = var.project_id

  uniform_bucket_level_access = true
  force_destroy               = var.environment == "dev" ? true : false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90 # Clean up old artifacts after 90 days
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 3 # Keep 3 versions max
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.apis]
}

# Grant the service account read/write access to the bucket
resource "google_storage_bucket_iam_member" "mlops_agent_storage" {
  bucket = google_storage_bucket.mlops_staging.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.mlops_agent.email}"
}

output "staging_bucket" {
  value = "gs://${google_storage_bucket.mlops_staging.name}"
}
