
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
WITH co AS (
    SELECT
        subject_id
        , hadm_id
        , stay_id
        , intime AS starttime
        , DATETIME_ADD(intime, INTERVAL '24' HOUR) AS endtime
    FROM `physionet-data.mimiciv_3_1_icu.icustays`
)

, cpap AS (
    SELECT
        co.subject_id
        , co.stay_id
        , GREATEST(
            MIN(DATETIME_SUB(charttime, INTERVAL '1' HOUR)), co.starttime
        ) AS starttime
        , LEAST(
            MAX(DATETIME_ADD(charttime, INTERVAL '4' HOUR)), co.endtime
        ) AS endtime
        , MAX(
            CASE
                WHEN
                    REGEXP_CONTAINS(LOWER(ce.value), '(cpap mask|bipap)') THEN 1
                ELSE 0
            END
        ) AS cpap
    FROM co
    INNER JOIN `physionet-data.mimiciv_3_1_icu.chartevents` ce
        ON co.stay_id = ce.stay_id
            AND ce.charttime > co.starttime
            AND ce.charttime <= co.endtime
    WHERE ce.itemid = 226732
        AND REGEXP_CONTAINS(LOWER(ce.value), '(cpap mask|bipap)')
    GROUP BY co.subject_id, co.stay_id, co.starttime, co.endtime
)

-- extract a flag for surgical service
-- this combined with "elective" from admissions table
-- defines elective/non-elective surgery
, surgflag AS (
    SELECT adm.hadm_id
        , CASE
            WHEN LOWER(curr_service) LIKE '%surg%' THEN 1 ELSE 0
        END AS surgical
        , ROW_NUMBER() OVER
        (
            PARTITION BY adm.hadm_id
            ORDER BY transfertime
        ) AS serviceorder
    FROM `physionet-data.mimiciv_3_1_hosp.admissions` adm
    LEFT JOIN `physionet-data.mimiciv_3_1_hosp.services` se
        ON adm.hadm_id = se.hadm_id
)

-- icd-9 diagnostic codes are our best source for comorbidity information
-- unfortunately, they are technically a-causal
-- however, this shouldn't matter too much for the SAPS II comorbidities
, comorb AS (
    SELECT hadm_id
        -- these are slightly different than elixhauser comorbidities,
        -- but based on them they include some non-comorbid ICD-9 codes
        -- (e.g. 20302, relapse of multiple myeloma)
        , MAX(CASE
            WHEN
                icd_version = 9 AND SUBSTR(
                    icd_code, 1, 3
                ) BETWEEN '042' AND '044'
                THEN 1
            WHEN
                icd_version = 10 AND SUBSTR(
                    icd_code, 1, 3
                ) BETWEEN 'B20' AND 'B22' THEN 1
            WHEN icd_version = 10 AND SUBSTR(icd_code, 1, 3) = 'B24' THEN 1
            ELSE 0 END) AS aids  /* HIV and AIDS */
        , MAX(
            CASE WHEN icd_version = 9 THEN
                CASE
                    -- lymphoma
                    WHEN
                        SUBSTR(
                            icd_code, 1, 5
                        ) BETWEEN '20000' AND '20238' THEN 1
                    -- leukemia
                    WHEN
                        SUBSTR(
                            icd_code, 1, 5
                        ) BETWEEN '20240' AND '20248' THEN 1
                    -- lymphoma
                    WHEN
                        SUBSTR(
                            icd_code, 1, 5
                        ) BETWEEN '20250' AND '20302' THEN 1
                    -- leukemia
                    WHEN
                        SUBSTR(
                            icd_code, 1, 5
                        ) BETWEEN '20310' AND '20312' THEN 1
                    -- lymphoma
                    WHEN
                        SUBSTR(
                            icd_code, 1, 5
                        ) BETWEEN '20302' AND '20382' THEN 1
                    -- chronic leukemia
                    WHEN
                        SUBSTR(
                            icd_code, 1, 5
                        ) BETWEEN '20400' AND '20522' THEN 1
                    -- other myeloid leukemia
                    WHEN
                        SUBSTR(
                            icd_code, 1, 5
                        ) BETWEEN '20580' AND '20702' THEN 1
                    -- other myeloid leukemia
                    WHEN
                        SUBSTR(
                            icd_code, 1, 5
                        ) BETWEEN '20720' AND '20892' THEN 1
                    -- lymphoma
                    WHEN SUBSTR(icd_code, 1, 4) IN ('2386', '2733') THEN 1
                    ELSE 0 END
                WHEN
                    icd_version = 10 AND SUBSTR(
                        icd_code, 1, 3
                    ) BETWEEN 'C81' AND 'C96' THEN 1
                ELSE 0 END) AS hem
        , MAX(CASE
            WHEN icd_version = 9 THEN
                CASE
                    WHEN SUBSTR(icd_code, 1, 4) BETWEEN '1960' AND '1991' THEN 1
                    WHEN
                        SUBSTR(
                            icd_code, 1, 5
                        ) BETWEEN '20970' AND '20975' THEN 1
                    WHEN SUBSTR(icd_code, 1, 5) IN ('20979', '78951') THEN 1
                    ELSE 0 END
            WHEN
                icd_version = 10 AND SUBSTR(
                    icd_code, 1, 3
                ) BETWEEN 'C77' AND 'C79' THEN 1
            WHEN icd_version = 10 AND SUBSTR(icd_code, 1, 4) = 'C800' THEN 1
            ELSE 0 END) AS mets      /* Metastatic cancer */
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    GROUP BY hadm_id
)

, pafi1 AS (
    -- join blood gas to ventilation durations to determine if patient was vent
    -- also join to cpap table for the same purpose
    SELECT
        co.stay_id
        , bg.charttime
        , pao2fio2ratio AS pao2fio2
        , CASE WHEN vd.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vent
        , CASE WHEN cp.subject_id IS NOT NULL THEN 1 ELSE 0 END AS cpap
    FROM co
    LEFT JOIN `physionet-data.mimiciv_3_1_derived.bg` bg
        ON co.subject_id = bg.subject_id
            AND bg.specimen = 'ART.'
            AND bg.charttime > co.starttime
            AND bg.charttime <= co.endtime
    LEFT JOIN `physionet-data.mimiciv_3_1_derived.ventilation` vd
        ON co.stay_id = vd.stay_id
            AND bg.charttime > vd.starttime
            AND bg.charttime <= vd.endtime
            AND vd.ventilation_status = 'InvasiveVent'
    LEFT JOIN cpap cp
        ON bg.subject_id = cp.subject_id
            AND bg.charttime > cp.starttime
            AND bg.charttime <= cp.endtime
)

, pafi2 AS (
    -- get the minimum PaO2/FiO2 ratio *only for ventilated/cpap patients*
    SELECT stay_id
        , MIN(pao2fio2) AS pao2fio2_vent_min
    FROM pafi1
    WHERE vent = 1 OR cpap = 1
    GROUP BY stay_id
)

, gcs AS (
    SELECT co.stay_id
        , MIN(gcs.gcs) AS mingcs
    FROM co
    LEFT JOIN `physionet-data.mimiciv_3_1_derived.gcs` gcs
        ON co.stay_id = gcs.stay_id
            AND co.starttime < gcs.charttime
            AND gcs.charttime <= co.endtime
    GROUP BY co.stay_id
)

, vital AS (
    SELECT
        co.stay_id
        , MIN(vital.heart_rate) AS heartrate_min
        , MAX(vital.heart_rate) AS heartrate_max
        , MIN(vital.sbp) AS sysbp_min
        , MAX(vital.sbp) AS sysbp_max
        , MIN(vital.temperature) AS tempc_min
        , MAX(vital.temperature) AS tempc_max
    FROM co
    LEFT JOIN `physionet-data.mimiciv_3_1_derived.vitalsign` vital
        ON co.subject_id = vital.subject_id
            AND co.starttime < vital.charttime
            AND co.endtime >= vital.charttime
    GROUP BY co.stay_id
)

, uo AS (
    SELECT
        co.stay_id
        , SUM(uo.urineoutput) AS urineoutput
    FROM co
    LEFT JOIN `physionet-data.mimiciv_3_1_derived.urine_output` uo
        ON co.stay_id = uo.stay_id
            AND co.starttime < uo.charttime
            AND co.endtime >= uo.charttime
    GROUP BY co.stay_id
)

, labs AS (
    SELECT
        co.stay_id
        , MIN(labs.bun) AS bun_min
        , MAX(labs.bun) AS bun_max
        , MIN(labs.potassium) AS potassium_min
        , MAX(labs.potassium) AS potassium_max
        , MIN(labs.sodium) AS sodium_min
        , MAX(labs.sodium) AS sodium_max
        , MIN(labs.bicarbonate) AS bicarbonate_min
        , MAX(labs.bicarbonate) AS bicarbonate_max
    FROM co
    LEFT JOIN `physionet-data.mimiciv_3_1_derived.chemistry` labs
        ON co.subject_id = labs.subject_id
            AND co.starttime < labs.charttime
            AND co.endtime >= labs.charttime
    GROUP BY co.stay_id
)

, cbc AS (
    SELECT
        co.stay_id
        , MIN(cbc.wbc) AS wbc_min
        , MAX(cbc.wbc) AS wbc_max
    FROM co
    LEFT JOIN `physionet-data.mimiciv_3_1_derived.complete_blood_count` cbc
        ON co.subject_id = cbc.subject_id
            AND co.starttime < cbc.charttime
            AND co.endtime >= cbc.charttime
    GROUP BY co.stay_id
)

, enz AS (
    SELECT
        co.stay_id
        , MIN(enz.bilirubin_total) AS bilirubin_min
        , MAX(enz.bilirubin_total) AS bilirubin_max
    FROM co
    LEFT JOIN `physionet-data.mimiciv_3_1_derived.enzyme` enz
        ON co.subject_id = enz.subject_id
            AND co.starttime < enz.charttime
            AND co.endtime >= enz.charttime
    GROUP BY co.stay_id
)

, cohort AS (
    SELECT
        ie.subject_id, ie.hadm_id, ie.stay_id,
        ie.first_careunit, ie.last_careunit
        , ie.intime
        , ie.outtime
        , CASE
        WHEN adm.deathtime >= ie.intime AND adm.deathtime <= ie.outtime THEN 1
        ELSE 0
      END AS icu_expire_flag -- 1 if died specifically in the ICU
      
        , va.age
        , co.starttime
        , co.endtime

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
        , cbc.wbc_min
        , cbc.wbc_max
        , labs.potassium_min
        , labs.potassium_max
        , labs.sodium_min
        , labs.sodium_max
        , labs.bicarbonate_min
        , labs.bicarbonate_max

        , enz.bilirubin_min
        , enz.bilirubin_max

        , gcs.mingcs

        , comorb.aids
        , comorb.hem
        , comorb.mets

        , CASE
            WHEN adm.admission_type = 'ELECTIVE' AND sf.surgical = 1
                THEN 'ScheduledSurgical'
            WHEN adm.admission_type != 'ELECTIVE' AND sf.surgical = 1
                THEN 'UnscheduledSurgical'
            ELSE 'Medical'
        END AS admissiontype


    FROM `physionet-data.mimiciv_3_1_icu.icustays` ie
    INNER JOIN `physionet-data.mimiciv_3_1_hosp.admissions` adm
        ON ie.hadm_id = adm.hadm_id
    LEFT JOIN `physionet-data.mimiciv_3_1_derived.age` va
        ON ie.hadm_id = va.hadm_id
    INNER JOIN co
        ON ie.stay_id = co.stay_id

    -- join to above views
    LEFT JOIN pafi2 pf
        ON ie.stay_id = pf.stay_id
    LEFT JOIN surgflag sf
        ON adm.hadm_id = sf.hadm_id AND sf.serviceorder = 1
    LEFT JOIN comorb
        ON ie.hadm_id = comorb.hadm_id

    -- join to custom tables to get more data....
    LEFT JOIN gcs gcs
        ON ie.stay_id = gcs.stay_id
    LEFT JOIN vital
        ON ie.stay_id = vital.stay_id
    LEFT JOIN uo
        ON ie.stay_id = uo.stay_id
    LEFT JOIN labs
        ON ie.stay_id = labs.stay_id
    LEFT JOIN cbc
        ON ie.stay_id = cbc.stay_id
    LEFT JOIN enz
        ON ie.stay_id = enz.stay_id
)

, scorecomp as (
  SELECT s.stay_id
    , s.starttime
    , s.endtime
    , s.*
FROM cohort s
)

-- Add the hospital mortality outcome

SELECT s.*, hospital_mortality,admittime, insurance, language, marital_status, race, anchor_age, anchor_year, anchor_year_group, gender, dod
FROM scorecomp s LEFT JOIN (
SELECT admittime, hadm_id, insurance, language, marital_status, race, hospital_expire_flag as hospital_mortality
FROM `physionet-data.mimiciv_3_1_hosp.admissions`) as death_tab
ON  s.hadm_id = death_tab.hadm_id
LEFT JOIN (SELECT 
        subject_id, anchor_age, anchor_year, anchor_year_group, gender, dod
    FROM `physionet-data.mimiciv_3_1_hosp.patients`
) as patient_info
  ON s.subject_id = patient_info.subject_id;