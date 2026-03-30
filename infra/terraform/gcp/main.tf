terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
  ])

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# Service account for mlops-agents workloads
resource "google_service_account" "mlops_agent" {
  account_id   = "mlops-agent-${var.environment}"
  display_name = "MLOps Agent Service Account (${var.environment})"
  project      = var.project_id
}

# Output the service account email for use in provider config
output "service_account_email" {
  value = google_service_account.mlops_agent.email
}

output "project_id" {
  value = var.project_id
}

output "region" {
  value = var.region
}
