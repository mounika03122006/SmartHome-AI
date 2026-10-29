import pytest
from src import auth


def test_create_and_verify(tmp_path):
    f = tmp_path / "a.json"
    assert not auth.is_configured(f)
    auth.create_admin("mounika", "StrongPass1", f)
    assert auth.is_configured(f)
    assert auth.verify("mounika", "StrongPass1", f)
    assert not auth.verify("mounika", "wrong", f)
    assert not auth.verify("other", "StrongPass1", f)
    assert "StrongPass1" not in f.read_text()  # password never stored in plain text


def test_single_admin_and_weak_password(tmp_path):
    f = tmp_path / "a.json"
    with pytest.raises(ValueError): auth.create_admin("x", "short", f)
    auth.create_admin("x", "longenough1", f)
    with pytest.raises(FileExistsError): auth.create_admin("y", "longenough2", f)


def test_change_password(tmp_path):
    f = tmp_path / "a.json"
    auth.create_admin("x", "longenough1", f)
    with pytest.raises(PermissionError): auth.change_password("bad", "newpassword1", f)
    auth.change_password("longenough1", "newpassword1", f)
    assert auth.verify("x", "newpassword1", f) and not auth.verify("x", "longenough1", f)


def test_verify_without_file(tmp_path):
    assert not auth.verify("x", "y", tmp_path / "none.json")
