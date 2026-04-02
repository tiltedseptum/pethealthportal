# Data Model Draft

## Pet
- id
- name
- species
- breed
- age
- weight
- conditions

## Timeline Event
- id
- pet_id
- event_type
- title
- description
- date

## Observation
- id
- pet_id
- notes
- category
- severity
- date
- image_url

## Preventive Care
- id
- pet_id
- type
- last_done
- next_due
- status

## Medical Record
- id
- pet_id
- type
- title
- date
- file_url
- notes

## Vet Visit
- id
- pet_id
- date
- clinic
- summary