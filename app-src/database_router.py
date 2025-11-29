class KeywordsRouter:
    """
    Un routeur pour diriger les opérations de base de données des modèles
    vers les bases de données appropriées.
    """

    def db_for_read(self, model, **hints):
        """
        Dirige les opérations de lecture pour les modèles vers les bases de données appropriées.
        """
        if model._meta.app_label == 'keywords_app':
            return 'keywords_db'
        elif model._meta.app_label == 'factiva_app':  # Application pour Factiva
            return 'factiva_db'
        return 'default'

    def db_for_write(self, model, **hints):
        """
        Dirige les opérations d'écriture pour les modèles vers les bases de données appropriées.
        """
        if model._meta.app_label == 'keywords_app':
            return 'keywords_db'
        elif model._meta.app_label == 'factiva_app':  # Application pour Factiva
            return 'factiva_db'
        return 'default'

    def allow_relation(self, obj1, obj2, **hints):
        """
        Permet les relations entre les objets des bases de données si elles appartiennent
        à la même base de données.
        """
        db_set = {'default', 'keywords_db', 'factiva_db'}
        if obj1._state.db in db_set and obj2._state.db in db_set:
            return True
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        Dirige les migrations pour les applications vers les bases de données appropriées.
        """
        if app_label == 'keywords_app':
            return db == 'keywords_db'
        elif app_label == 'factiva_app':  # Application pour Factiva
            return db == 'factiva_db'
        return db == 'default'