from modaic.storage.file_store import InPlaceFileStore

x = InPlaceFileStore("test_data")

print(x.id_to_files)
print(x.file_to_ids)

x._put(x.id_to_files, "test_id", {"test_alias": "test_path"})

print(x._get(x.id_to_files, "test_id"))
print(x._get(x.file_to_ids, "not_in_there"))
