def test_healthcheck(set_env_var, test_app):
    response = test_app.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
