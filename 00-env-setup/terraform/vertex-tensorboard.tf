# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

module tensorboard {
  source  = "terraform-google-modules/gcloud/google"
  version = "~> 2.0"

  platform = "linux"

  create_cmd_entrypoint  = "printf 'yes' | gcloud"
  create_cmd_body        = "beta ai tensorboards create --display-name ${var.name_prefix}-${var.subnet_region}-tensorboard --project ${var.project_id} --region ${var.subnet_region}"
  destroy_cmd_entrypoint = "printf 'yes' | gcloud"
  destroy_cmd_body       = "beta ai tensorboards delete $(gcloud beta ai tensorboards list --region ${var.subnet_region} --filter='displayName:${var.name_prefix}-${var.subnet_region}-tensorboard' --format='value(name)' --project ${var.project_id})"
  
  depends_on = [module.project-services.api_activated]
}