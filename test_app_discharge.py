from streamlit.testing.v1 import AppTest
import os
at = AppTest.from_file("app_discharge.py", default_timeout=120).run()
print("1 exceptions:", [e.value for e in at.exception], "| tabs:", len(at.tabs), "| title:", at.title[0].value)
print("  metrics:", [(m.label, m.value) for m in at.metric])
print("  dataframes:", len(at.dataframe), "| tables:", len(at.table))
# switch to a synthetic test-set note and shrink the summary
at.sidebar.radio[0].set_value("Test-set note").run()
print("2 exceptions:", [e.value for e in at.exception], "| selectbox options:", len(at.sidebar.selectbox[0].options))
at.sidebar.selectbox[0].select("note_05").run(); at.sidebar.slider[0].set_value(2).run()
print("3 exceptions:", [e.value for e in at.exception], "| summary bullets:", len([m for m in at.markdown if m.value.startswith("- ")]))
# paste mode with empty text, then with text
at.sidebar.radio[0].set_value("Paste text").run(); print("4 empty paste info:", [i.value for i in at.info])
at.sidebar.text_area[0].set_value("Hospital Course: He was started on aspirin. Pain resolved.\nDischarge Medications:\n1. Aspirin 81 mg PO daily (NEW)").run()
print("5 exceptions:", [e.value for e in at.exception], "| meds table rows:", len(at.dataframe[0].value))
