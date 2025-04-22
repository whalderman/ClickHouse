#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypeFixedString.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypesNumber.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnArray.h>
#include <Interpreters/Context_fwd.h>
#include <Interpreters/ITokenExtractor.h>
#include <Functions/IFunction.h>
#include <Functions/FunctionHelpers.h>
#include <Functions/FunctionFactory.h>
#include "Interpreters/GinFilter.h"

namespace DB
{

namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
}

struct GinMatchAnyProps
{
    static constexpr auto name = "ginMatchAny";
    static constexpr auto match = GinMatch::ANY;
};

struct GinMatchAllProps
{
    static constexpr auto name = "ginMatchAll";
    static constexpr auto match = GinMatch::ALL;
};

template <typename GinMatchProps>
class FunctionGinMatchImpl : public IFunction
{
public:
    static constexpr auto name = GinMatchProps::name;

    static FunctionPtr create(ContextPtr)
    {
        return std::make_shared<FunctionGinMatchImpl>();
    }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 3; }
    bool isVariadic() const override { return true; }
    ColumnNumbers getArgumentsThatAreAlwaysConstant() const override { return {}; }

    bool useDefaultImplementationForNulls() const override { return true; }
    bool useDefaultImplementationForConstants() const override { return true; }
    bool useDefaultImplementationForLowCardinalityColumns() const override { return true; }
    bool isSuitableForShortCircuitArgumentsExecution(const DataTypesWithConstInfo & /*arguments*/) const override { return true; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        if (arguments.size() < 3 || arguments.size() > 4) {
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Function {} argument count does not match", getName());
        }
        if (!WhichDataType(arguments[0].type).isStringOrFixedString())
            throw Exception(ErrorCodes::BAD_ARGUMENTS,
                "Function {} first argument type should be String or FixedString. Actual {}",
                getName(),
                arguments[0].type->getName());

        if (arguments.size() == 3)
        {
            if (!WhichDataType(arguments[1].type).isStringOrFixedString())
                throw Exception(
                    ErrorCodes::BAD_ARGUMENTS,
                    "Function {} second argument type should be Column name. Actual {}",
                    getName(),
                    arguments[1].type->getName());

            if (!WhichDataType(arguments[2].type).isStringOrFixedString())
                throw Exception(
                    ErrorCodes::BAD_ARGUMENTS,
                    "Function {} third argument type should be String or FixedString. Actual {}",
                    getName(),
                    arguments[2].type->getName());
        }
        else if (arguments.size() == 4)
        {
            if (!WhichDataType(arguments[1].type).isUInt())
                throw Exception(
                    ErrorCodes::BAD_ARGUMENTS,
                    "Function {} second argument type should be Integer. Actual {}",
                    getName(),
                    arguments[1].type->getName());

            if (!WhichDataType(arguments[2].type).isStringOrFixedString())
                throw Exception(
                    ErrorCodes::BAD_ARGUMENTS,
                    "Function {} third argument type should be Column name. Actual {}",
                    getName(),
                    arguments[2].type->getName());

            if (!WhichDataType(arguments[3].type).isStringOrFixedString())
                throw Exception(
                    ErrorCodes::BAD_ARGUMENTS,
                    "Function {} fourth argument type should be String or FixedString. Actual {}",
                    getName(),
                    arguments[3].type->getName());
        }

        return std::make_shared<DataTypeNumber<UInt8>>();
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        auto input_column = arguments.size() == 3 ? arguments[1].column : arguments[2].column;
        auto token_column = arguments.size() == 3 ? arguments[2].column : arguments[3].column;

        const auto * tokenizer_col{checkAndGetColumn<ColumnConst>(arguments[0].column.get())};
        if (tokenizer_col == nullptr)
            throw Exception(
                ErrorCodes::BAD_ARGUMENTS,
                "Function {} first argument type should be const String. Actual {}",
                getName(),
                arguments[0].type->getName());

        const auto * token_col{checkAndGetColumn<ColumnConst>(token_column.get())};
        if (token_col == nullptr)
            throw Exception(
                ErrorCodes::BAD_ARGUMENTS,
                "Function {} third argument type should be const String. Actual {}",
                getName(),
                arguments[3].type->getName());

        auto token_extractor{
            [&arguments, tokenizer = tokenizer_col->getDataAt(0).toString()] -> std::unique_ptr<ITokenExtractor>
            {
                if (tokenizer == GIN_TOKENIZER_DEFAULT)
                    return std::make_unique<SplitTokenExtractor>();
                else if (tokenizer == GIN_TOKENIZER_NONE)
                    return std::make_unique<NoneTokenExtractor>();
                if (tokenizer == GIN_TOKENIZER_CHINESE)
                    return std::make_unique<ChineseTokenExtractor>();
                if (tokenizer == GIN_TOKENIZER_NGRAM)
                {
                    const auto * ngram_param_col{checkAndGetColumn<ColumnConst>(arguments[1].column.get())};
                    return std::make_unique<NgramTokenExtractor>(ngram_param_col->getUInt(0));
                }
                throw Exception(
                    ErrorCodes::BAD_ARGUMENTS, "Only 'whitespace' or 'chinese' tokenizers are allowed, but got '{}'", tokenizer);
            }()};

        auto result_column = ColumnVector<UInt8>::create();

        SplitTokenExtractor needle_token_extractor{};
        std::unordered_set<std::string> needle_tokens{};
        const auto & needle_token_col{token_col->getDataAt(0)};
        for (auto needle_token : needle_token_extractor.getTokens(needle_token_col.data, needle_token_col.size))
            needle_tokens.emplace(std::move(needle_token));

        if (const auto * column_string = checkAndGetColumn<ColumnString>(input_column.get()))
            executeImpl(std::move(token_extractor), *column_string, input_rows_count, needle_tokens, result_column->getData());
        else if (const auto * column_fixed_string = checkAndGetColumn<ColumnFixedString>(input_column.get()))
            executeImpl(std::move(token_extractor), *column_fixed_string, input_rows_count, needle_tokens, result_column->getData());
        else
            throw Exception(
                ErrorCodes::BAD_ARGUMENTS,
                "Function {} second argument type should be string column. Actual {}",
                getName(),
                arguments[1].type->getName());

        return result_column;
    }

private:
    template <typename StringColumnType>
    void executeImpl(
        std::unique_ptr<ITokenExtractor> token_extractor,
        StringColumnType & input_column,
        size_t input_rows_count,
        const std::unordered_set<std::string>& needle_tokens,
        PaddedPODArray<UInt8> & result) const
    {
        result.resize(input_rows_count);

        for (size_t i = 0; i < input_rows_count; ++i)
        {
            const auto value{input_column.getDataAt(i)};

            result[i] = false;

            [[maybe_unused]] UInt64 mask{};
            for (const auto& token : token_extractor->getTokens(value.data, value.size))
            {
                if (auto it{needle_tokens.find(token)}; it != needle_tokens.end()) {
                    if constexpr (GinMatchProps::match == GinMatch::ALL)
                    {
                        auto dist{std::distance(needle_tokens.begin(), it)};
                        mask |= 1 << dist;
                    }
                    else
                    {
                        result[i] = true;
                        break;
                    }
                }
            }
            if constexpr (GinMatchProps::match == GinMatch::ALL)
            {
                result[i] = mask == ((1 << needle_tokens.size()) - 1);
            }
        }
    }
};

REGISTER_FUNCTION(GinMatchAny)
{
    factory.registerFunction<FunctionGinMatchImpl<GinMatchAnyProps>>();
}

REGISTER_FUNCTION(GinMatchAll)
{
    factory.registerFunction<FunctionGinMatchImpl<GinMatchAllProps>>();
}

}


