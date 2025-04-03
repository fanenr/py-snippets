import sqlite3
from nicegui import ui

conn = sqlite3.connect("one-api.db")
conn.row_factory = sqlite3.Row

with ui.header().classes("items-center"):
    ui.button(on_click=lambda: drawer.toggle(), icon="menu").props("flat color=white")
    ui.label("OneAPI Tool").classes("text-xl text-bold")

with ui.left_drawer(value=False).props("bordered") as drawer:
    with ui.list().props("bordered seprator").classes("w-full"):
        with ui.item(on_click=lambda: page_switch(table)):
            with ui.item_section().props("avatar"):
                ui.icon("table_chart", color="primary")
            with ui.item_section():
                ui.label("Channels")
        with ui.item(on_click=lambda: page_switch(info)):
            with ui.item_section().props("avatar"):
                ui.icon("analytics", color="primary")
            with ui.item_section():
                ui.label("Information")


cursor = conn.cursor()
rows = cursor.execute("SELECT * FROM channels;").fetchall()

cols = [
    {"name": "id", "label": "ID", "field": "id"},
    {"name": "name", "label": "Name", "field": "name"},
    {"name": "url", "label": "BaseURL", "field": "url"},
    {"name": "key", "label": "APIKey", "field": "key"},
    {"name": "used", "label": "Used", "field": "used"},
    {"name": "rest", "label": "Rest", "field": "rest"},
    {"name": "speed", "label": "Speed", "field": "speed"},
]

params = {
    "rows": [],
    "columns": cols,
    "column_defaults": {
        "align": "left",
        "sortable": True,
        "headerClasses": "text-primary",
    },
}

with ui.table(**params).props("flat bordered").classes("w-full") as table:
    for row in rows:
        used = row["used_quota"] / 500000
        speed = f"{row['response_time'] / 1000:.2f}s"
        rest = row["balance"] - used if row["balance"] else 0

        table.add_row(
            {
                "used": used,
                "rest": rest,
                "speed": speed,
                "id": row["id"],
                "key": row["key"],
                "name": row["name"],
                "url": row["base_url"],
            }
        )

with ui.card().props("flat bordered").classes("w-full") as info:
    ui.label("Channel Information").classes("text-base text-bold")

    def on_change(e):
        row = next((r for r in rows if r["name"] == e.value))
        edit_mapping.set_value(row["model_mapping"])
        edit_models.set_value(row["models"])
        edit_url.set_value(row["base_url"])
        edit_name.set_value(row["name"])
        edit_key.set_value(row["key"])

    with ui.row().classes("w-full justify-center"):
        with ui.column():
            ui.label("Channel")
            ui.select(
                options=[r["name"] for r in rows], on_change=on_change, with_input=True
            ).props("outlined dense")
        with ui.column():
            ui.label("Channel Nmae")
            edit_name = ui.input().props("outlined dense")
        with ui.column():
            ui.label("Base URL")
            edit_url = ui.input().props("outlined dense")
        with ui.column():
            ui.label("API Key")
            edit_key = ui.input().props("outlined dense")

    with ui.row().classes("w-full justify-center"):
        with ui.column().classes("w-1/3"):
            ui.label("Models")
            edit_models = ui.textarea().props("outlined dense").classes("w-full")
        with ui.column().classes("w-1/3"):
            ui.label("Model Mapping")
            edit_mapping = ui.textarea().props("outlined dense").classes("w-full")


def page_switch(page):
    [p.set_visibility(False) for p in pages]
    page.set_visibility(True)


pages = [table, info]
page_switch(table)

ui.run(port=8888)
