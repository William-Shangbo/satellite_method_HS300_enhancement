from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from index_platform.common.settings import RuntimeSettings


class TushareAPIError(RuntimeError):
    """Raised when Tushare returns a non-zero business code."""


@dataclass(frozen=True)
class TushareRequest:
    api_name: str
    params: dict[str, Any]
    fields: tuple[str, ...] = ()

    def to_payload(self, token: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "api_name": self.api_name,
            "token": token,
            "params": self.params,
        }
        if self.fields:
            payload["fields"] = ",".join(self.fields)
        return payload


class TushareHTTPClient:
    def __init__(
        self,
        settings: RuntimeSettings,
        endpoint: str = "http://api.tushare.pro",
        timeout_seconds: int = 30,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def fetch_frame(self, request: TushareRequest) -> pd.DataFrame:
        if not self.settings.tushare_token:
            raise ValueError("Missing TUSHARE_TOKEN in runtime settings.")

        response = self.session.post(
            self.endpoint,
            json=request.to_payload(self.settings.tushare_token),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise TushareAPIError(
                f"Tushare api_name={request.api_name!r} failed with "
                f"code={payload.get('code')}, msg={payload.get('msg')}"
            )
        fields = payload["data"]["fields"]
        items = payload["data"]["items"]
        return pd.DataFrame(items, columns=fields)
