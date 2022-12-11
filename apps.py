from django.apps import AppConfig
import bsbi as bsbi
import compression as compression

class MedexConfig(AppConfig):
    name = 'medex'
    verbose_name = "Medex"
    def ready(self):
        print("AA READY")
        BSBI_instance = bsbi.BSBIIndex(data_dir='collection',
                                       postings_encoding=compression.VBEPostings,
                                       output_dir='index')
        BSBI_instance.index()