SELECT SUBSTR(
            DOC_CONTENT,
            INSTR(LOWER(DOC_CONTENT), 'acknowledgements') + LENGTH('acknowledgements'),
            INSTR(LOWER(DOC_CONTENT), 'references') - (INSTR(LOWER(DOC_CONTENT), 'acknowledgements') + LENGTH('Acknowledgements'))
        ) AS content_between_ack_and_dest
FROM mspubs.manuscript_realtime_document
WHERE INSTR(LOWER(DOC_CONTENT), 'acknowledgements') > 0 AND INSTR(lower(DOC_CONTENT), 'references') > 0
and manuscript_number = 'am-2022-01288a' and doc_type = 'SUBMISSION_FILE'
AND revision = (
  SELECT MAX(revision)
  FROM mspubs.manuscript_realtime_document
  WHERE manuscript_number =  'am-2022-01288a'
);