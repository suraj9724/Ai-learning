import re


class CitationValidator:

    CITATION_PATTERN = re.compile(
        r"\[Source ID:\s*(\d+)\]"
    )

    def validate(
        self,
        answer: str,
        results: list[dict],
    ):

        cited_ids = {
            int(match)
            for match in self.CITATION_PATTERN.findall(
                answer
            )
        }

        available_ids = {
            result["chunk"]["chunk_id"]
            for result in results
        }

        invalid_ids = (
            cited_ids - available_ids
        )

        return {
            "valid": len(invalid_ids) == 0,
            "cited_ids": sorted(cited_ids),
            "invalid_ids": sorted(invalid_ids),
        }