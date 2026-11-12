from text2image.cli import build_parser, main


def test_generate_arguments_parse_without_loading_runtime():
    args = build_parser().parse_args(
        ["generate", "--model", "flux-schnell", "--prompt", "a red fox", "--seed", "7"]
    )
    assert args.command == "generate"
    assert args.model == "flux-schnell"
    assert args.prompt == "a red fox"
    assert args.seed == 7


def test_session_arguments_parse_without_loading_runtime():
    args = build_parser().parse_args(
        ["session", "--model", "sdxl", "--output-dir", "renders", "--steps", "5"]
    )
    assert args.command == "session"
    assert args.model == "sdxl"
    assert str(args.output_dir) == "renders"
    assert args.steps == 5


def test_list_command_does_not_import_or_download_models(capsys):
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "FLUX.1 [dev]" in output
    assert "Best for:" in output
    assert "Main trade-off:" in output
