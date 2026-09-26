
-- ------------------------------------------------------------------
-- Title: Simplified Acute Physiology Score II (SAPS II)
-- This query extracts the simplified acute physiology score II.
-- This score is a measure of patient severity of illness.
-- The score is calculated on the first day of each ICU patients' stay.
-- ------------------------------------------------------------------

-- Reference for SAPS II:
--    Le Gall, Jean-Roger, Stanley Lemeshow, and Fabienne Saulnier.
--    "A new simplified acute physiology score (SAPS II) based on
--    a European/North American multicenter study."
--    JAMA 270, no. 24 (1993): 2957-2963.

-- Variables used in SAPS II:
--  Age, GCS
--  VITALS: Heart rate, systolic blood pressure, temperature
--  FLAGS: ventilation/cpap
--  IO: urine output
--  LABS: PaO2/FiO2 ratio, blood urea nitrogen, WBC,
--      potassium, sodium, HCO3

with cpap as
(
  select ie.icustay_id
    , min(DATETIME_SUB(charttime, INTERVAL '1' HOUR)) as starttime
    , max(DATETIME_ADD(charttime, INTERVAL '4' HOUR)) as endtime
    , max(CASE
          WHEN lower(ce.value) LIKE '%cpap%' THEN 1
          WHEN lower(ce.value) LIKE '%bipap mask%' THEN 1
        else 0 end) as cpap
  FROM `physionet-data.mimiciii_clinical.icustays` ie
  inner join `physionet-data.mimiciii_clinical.chartevents` ce
    on ie.icustay_id = ce.icustay_id
    and ce.charttime between ie.intime and DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
  where itemid in
  (
    467, 469, 226732
  )
  and (lower(ce.value) LIKE '%cpap%' or lower(ce.value) LIKE '%bipap mask%')
  -- exclude rows marked as error
  AND (ce.error IS NULL OR ce.error = 0)
  group by ie.icustay_id
)
-- extract a flag for surgical service
-- this combined with "elective" FROM `physionet-data.mimiciii_clinical.admissions` table defines elective/non-elective surgery
, surgflag as
(
  select adm.hadm_id
    , case when lower(curr_service) like '%surg%' then 1 else 0 end as surgical
    , ROW_NUMBER() over
    (
      PARTITION BY adm.HADM_ID
      ORDER BY TRANSFERTIME
    ) as serviceOrder
  FROM `physionet-data.mimiciii_clinical.admissions` adm
  left join `physionet-data.mimiciii_clinical.services` se
    on adm.hadm_id = se.hadm_id
)
, comorb as
(
select hadm_id
  , max(CASE
    when SUBSTR(icd9_code,1,3) BETWEEN '042' AND '044' THEN 1
  		end) as aids      /* HIV and AIDS */
  , max(CASE
    when icd9_code between '20000' and '20238' then 1 -- lymphoma
    when icd9_code between '20240' and '20248' then 1 -- leukemia
    when icd9_code between '20250' and '20302' then 1 -- lymphoma
    when icd9_code between '20310' and '20312' then 1 -- leukemia
    when icd9_code between '20302' and '20382' then 1 -- lymphoma
    when icd9_code between '20400' and '20522' then 1 -- chronic leukemia
    when icd9_code between '20580' and '20702' then 1 -- other myeloid leukemia
    when icd9_code between '20720' and '20892' then 1 -- other myeloid leukemia
    when SUBSTR(icd9_code,1,4) = '2386' then 1 -- lymphoma
    when SUBSTR(icd9_code,1,4) = '2733' then 1 -- lymphoma
  		end) as hem
  , max(CASE
    when SUBSTR(icd9_code,1,4) BETWEEN '1960' AND '1991' THEN 1
    when icd9_code between '20970' and '20975' then 1
    when icd9_code = '20979' then 1
    when icd9_code = '78951' then 1
  		end) as mets      /* Metastatic cancer */
  from `physionet-data.mimiciii_clinical.diagnoses_icd`
  group by hadm_id
)
, pafi1 as
(
  -- join blood gas to ventilation durations to determine if patient was vent
  -- also join to cpap table for the same purpose
  select bg.icustay_id, bg.charttime
  , pao2fio2
  , case when vd.icustay_id is not null then 1 else 0 end as vent
  , case when cp.icustay_id is not null then 1 else 0 end as cpap
  from `physionet-data.mimiciii_derived.blood_gas_first_day_arterial` bg
  left join `physionet-data.mimiciii_derived.ventilation_durations` vd
    on bg.icustay_id = vd.icustay_id
    and bg.charttime >= vd.starttime
    and bg.charttime <= vd.endtime
  left join cpap cp
    on bg.icustay_id = cp.icustay_id
    and bg.charttime >= cp.starttime
    and bg.charttime <= cp.endtime
)
, pafi2 as
(
  -- get the minimum PaO2/FiO2 ratio *only for ventilated/cpap patients*
  select icustay_id
  , min(pao2fio2) as pao2fio2_vent_min
  from pafi1
  where vent = 1 or cpap = 1
  group by icustay_id
)
, cohort as
(
select ie.subject_id, ie.hadm_id, ie.icustay_id, ie.first_careunit, ie.last_careunit
      , ie.intime
      , ie.outtime
      , pat.DOD
      , pat.DOD_HOSP
      , pat.DOD_SSN
      , pat.DOB
      , adm.deathtime
      , adm.hospital_expire_flag -- 1 if died anytime in hospital
    , CASE
        WHEN adm.deathtime >= ie.intime AND adm.deathtime <= ie.outtime THEN 1
        ELSE 0
      END AS icu_expire_flag -- 1 if died specifically in the ICU
      
      -- the casts ensure the result is numeric.. we could equally extract EPOCH from the interval
      -- however this code works in Oracle and Postgres
      , DATETIME_DIFF(ie.intime, pat.dob, YEAR) as age

      , vital.heartrate_max
      , vital.heartrate_min
      , vital.sysbp_max
      , vital.sysbp_min
      , vital.tempc_max
      , vital.tempc_min

      -- this value is non-null iff the patient is on vent/cpap
      , pf.pao2fio2_vent_min

      , uo.urineoutput

      , labs.bun_min
      , labs.bun_max
      , labs.wbc_min
      , labs.wbc_max
      , labs.potassium_min
      , labs.potassium_max
      , labs.sodium_min
      , labs.sodium_max
      , labs.bicarbonate_min
      , labs.bicarbonate_max
      , labs.bilirubin_min
      , labs.bilirubin_max

      , gcs.mingcs

      , comorb.aids
      , comorb.hem
      , comorb.mets

      , case
          when adm.ADMISSION_TYPE = 'ELECTIVE' and sf.surgical = 1
            then 'ScheduledSurgical'
          when adm.ADMISSION_TYPE != 'ELECTIVE' and sf.surgical = 1
            then 'UnscheduledSurgical'
          else 'Medical'
        end as admissiontype


FROM `physionet-data.mimiciii_clinical.icustays` ie
inner join `physionet-data.mimiciii_clinical.admissions` adm
  on ie.hadm_id = adm.hadm_id
inner join `physionet-data.mimiciii_clinical.patients` pat
  on ie.subject_id = pat.subject_id

-- join to above views
left join pafi2 pf
  on ie.icustay_id = pf.icustay_id
left join surgflag sf
  on adm.hadm_id = sf.hadm_id and sf.serviceOrder = 1
left join comorb
  on ie.hadm_id = comorb.hadm_id

-- join to custom tables to get more data....
left join `physionet-data.mimiciii_derived.gcs_first_day` gcs
  on ie.icustay_id = gcs.icustay_id
left join `physionet-data.mimiciii_derived.vitals_first_day` vital
  on ie.icustay_id = vital.icustay_id
left join `physionet-data.mimiciii_derived.urine_output_first_day` uo
  on ie.icustay_id = uo.icustay_id
left join `physionet-data.mimiciii_derived.labs_first_day` labs
  on ie.icustay_id = labs.icustay_id
)
, scorecomp as
(
select
  cohort.*
  -- Raw features are already contained in cohort 
from cohort
)

-- Add the hospital mortality outcome
SELECT scorecomp.*, INSURANCE, marital_status, ethnicity, language, hospital_mortality, GENDER as gender, EXPIRE_FLAG as death
FROM scorecomp LEFT JOIN (
SELECT hadm_id, MARITAL_STATUS, ETHNICITY, LANGUAGE, INSURANCE, hospital_expire_flag as hospital_mortality
FROM `physionet-data.mimiciii_clinical.admissions`) as death_tab
ON  scorecomp.hadm_id = death_tab.hadm_id
LEFT JOIN (
SELECT gender, SUBJECT_ID, EXPIRE_FLAG
FROM `physionet-data.mimiciii_clinical.patients`) as pat_tab
ON scorecomp.subject_id = pat_tab.SUBJECT_ID