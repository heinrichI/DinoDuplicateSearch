import flet as ft
import asyncio

async def main(page: ft.Page):
    page.title = "Диалог с прогрессом"
    page.window.width = 400
    page.window.height = 250

    pb = ft.ProgressBar(width=300, value=0)
    dlg = ft.AlertDialog(
        modal=True,
        title=ft.Text("Пожалуйста, подождите..."),
        content=pb,
        actions=[],
    )

    async def start_task(e):
        print("Кнопка нажата, открываем диалог")  # отладка
        page.show_dialog(dlg)
        print("Диалог должен быть виден")         # отладка

        for i in range(1, 11):
            await asyncio.sleep(0.5)
            pb.value = i / 10
            page.update()

        page.pop_dialog()
        print("Диалог закрыт")

    page.add(
        ft.Column(
            [
                ft.Text("Нажмите кнопку, чтобы увидеть диалог"),
                ft.ElevatedButton("Запустить", on_click=start_task),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
    )

ft.app(target=main)